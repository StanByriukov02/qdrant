use std::hash::{BuildHasher, Hash};
use std::sync::atomic::{AtomicU32, Ordering};

use hashbrown::HashTable;
use parking_lot::Mutex;

use crate::s3fifo::{Entry, LocalOffset, S3Fifo};
use crate::seqlock::{SeqLock, SeqLockReader, SeqLockWriter};

/// A concurrent S3-FIFO cache with a `hashbrown::HashTable` which stores the
/// keys in the FIFO queues for lower memory overhead.
///
/// Design:
/// - Reads are lock-free (only atomic seqlock checks) and can happen concurrently.
/// - Writers share the same writer behind a mutex, so they will contend if another
///   write is taking place.
pub struct Cache<K, V, S = ahash::RandomState> {
    /// Shared state for lock-free readers.
    reader: SeqLockReader<CacheInner<K, V, S>>,

    writer: Mutex<SeqLockWriter<CacheInner<K, V, S>>>,
}

/// Inner cache state. This is the state the seqlock protects.
/// Readers observe an immutable snapshot of this struct via `SeqLockReader`.
struct CacheInner<K, V, S> {
    /// Non-concurrent hashtable mapping key -> global offset in the FIFOs.
    hashtable: HashTable<AtomicU32>,
    /// The actual FIFO structures (small, ghost, main).
    fifos: S3Fifo<K, V>,
    /// Hasher state used to compute the hash for lookups.
    hasher: S,
}

impl<K, V> Cache<K, V, ahash::RandomState>
where
    K: Default + Copy + Hash + Eq + Send + Sync + 'static,
    V: Default + Clone + Send + Sync + 'static,
{
    /// Create a new concurrent cache with default disruptor size and producer pool.
    pub fn new(capacity: usize) -> Self {
        Self::with_config(capacity, 0.1, 0.9, ahash::RandomState::new())
    }
}

impl<K, V, S> Cache<K, V, S>
where
    K: Default + Copy + Hash + Eq + Send + Sync + 'static,
    V: Default + Clone + Send + Sync + 'static,
    S: BuildHasher + Clone + Default + Send + Sync + 'static,
{
    /// Create with custom disruptor ring size.
    pub fn with_config(capacity: usize, small_ratio: f32, ghost_ratio: f32, hasher: S) -> Self {
        assert!(capacity > 0);

        // Create FIFOs (reader + writer halves are managed by S3Fifo)
        let fifos = S3Fifo::new(capacity, small_ratio, ghost_ratio);

        // Create a hashbrown hashtable with requested capacity.
        //
        // Maximum entries = small + main + ghost = capacity + ghost_size
        let max_entries = capacity + (capacity as f32 * ghost_ratio) as usize;

        // IMPORTANT: Allocate for 2x as much entries so that hashtable can be rehashed in-place if needed,
        // but never resized and reallocate.
        // See: https://github.com/rust-lang/hashbrown/blob/9641fb3eea9a07933fb631da6e4f5070d2f7e1da/src/raw.rs#L2775
        //
        // Internally, the capacity increases to 1/8 higher, but that should only make it more robust.
        // By doing this, we ensure that concurrent readers will not get to a use-after-free error.
        let table_capacity = max_entries * 2;
        let hashtable = HashTable::with_capacity(table_capacity);

        let cache_inner = CacheInner {
            hashtable,
            fifos,
            hasher,
        };

        // Create seqlock reader/writer pair
        let (reader, cache_writer) = SeqLock::new_reader_writer(cache_inner);

        Self {
            reader,
            writer: Mutex::new(cache_writer),
        }
    }

    /// Lock-free read of a value from the cache.
    ///
    /// Returns a cloned value (if present and not in the ghost queue).
    #[inline]
    pub fn get(&self, key: &K) -> Option<V> {
        self.reader.read(|cache| {
            // Compute hash using the stored hasher
            let hash = cache.hash_key(key);

            let global_offset = cache.hashtable.find(hash, |global_offset| {
                cache
                    .fifos
                    .key_eq(global_offset.load(Ordering::Relaxed), key)
            })?;

            let local = cache
                .fifos
                .local_offset(global_offset.load(Ordering::Relaxed));
            let entry = cache.fifos.get_entry(local)?;
            entry.incr_recency();
            Some(entry.value.clone())
        })
    }

    /// Publish an insert for processing by the writer thread.
    #[inline]
    pub fn insert(&self, key: K, value: V) {
        let writer_guard = self.writer.lock();
        unsafe { writer_guard.write(|cache| cache.do_insert(key, value)) };
    }

    /// Number of entries tracked by the hashtable (reads via seqlock).
    pub fn len(&self) -> usize {
        self.reader.read(|cache| cache.hashtable.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ============================================================================
// CacheInner: writer-side logic (only touched by writer thread + seqlock)
// ============================================================================

impl<K, V, S> CacheInner<K, V, S>
where
    K: Copy + Hash + Eq + Send + Sync,
    V: Clone + Send + Sync,
    S: BuildHasher + Clone + Default + Send + Sync,
{
    /// Convenience to compute hash for a key using stored hasher.
    #[inline]
    fn hash_key(&self, key: &K) -> u64 {
        // Reuse same helper used in `cache.rs` / other modules: BuildHasher impls
        // in this crate expose `hash_one`. If unavailable, adapt accordingly.
        self.hasher.hash_one(key)
    }

    /// Update hashtable entry for `key`. If it did not exist, does nothing.
    fn update_hashtable(&mut self, key: &K, local_offset: LocalOffset) {
        let hash = self.hash_key(key);

        let global_offset = self.fifos.global_offset(local_offset);
        // Use the find_entry API to update.
        let entry = self.hashtable.find_entry(hash, |global_offset| {
            self.fifos
                .key_eq(global_offset.load(Ordering::Relaxed), key)
        });

        if let Ok(occupied) = entry {
            occupied.get().store(global_offset, Ordering::Relaxed);
        }
    }

    /// Remove a key from the hashtable if present.
    #[inline]
    fn remove_from_hashtable(&mut self, key: &K) {
        let hash = self.hash_key(key);
        if let Ok(entry) = self.hashtable.find_entry(hash, |global_offset| {
            self.fifos
                .key_eq(global_offset.load(Ordering::Relaxed), key)
        }) {
            entry.remove();
        }
    }

    /// Insert a fresh entry into the hashtable. Use this when we've already
    /// removed the old entry and need to insert at a new location.
    /// This avoids the key_eq lookup which can fail if the old slot was overwritten.
    fn insert_unique_to_hashtable(&mut self, key: &K, local_offset: LocalOffset) {
        let hash = self.hash_key(key);
        let global_offset = self.fifos.global_offset(local_offset);

        // Insert directly without searching for existing entry.
        // Caller must ensure the old entry was already removed.
        self.hashtable
            .insert_unique(hash, AtomicU32::new(global_offset), |global_offset| {
                self.fifos
                    .hash_key_at_offset(global_offset.load(Ordering::Relaxed), &self.hasher)
            });
    }

    /// Main insert implementation. Only executed on writer thread.
    fn do_insert(&mut self, key: K, value: V) {
        // Check existing entry
        let hash = self.hash_key(&key);
        if let Some(global_offset) = self.hashtable.find(hash, |global_offset| {
            self.fifos
                .key_eq(global_offset.load(Ordering::Relaxed), &key)
        }) {
            let local = self
                .fifos
                .local_offset(global_offset.load(Ordering::Relaxed));
            self.promote_existing(local, key, value);
            return;
        }

        // New entry -> insert into small queue
        let entry = Entry::new(key, value);
        let local = self.push_to_small_queue(entry);
        self.insert_unique_to_hashtable(&key, local);
    }

    /// Promote existing entry or increment recency.
    fn promote_existing(&mut self, local_offset: LocalOffset, key: K, value: V) {
        match local_offset {
            LocalOffset::Ghost(offset) => {
                // Verify ghost key still matches
                let ghost_key = self.fifos.get_ghost_key(offset);
                if ghost_key != &key {
                    debug_assert!(false, "Key mismatch in ghost promotion");
                    return;
                }

                let entry = Entry::new(key, value);
                let new_local = self.push_to_main_queue(entry);
                self.insert_unique_to_hashtable(&key, new_local);
            }
            LocalOffset::Small(offset) => {
                let entry = self.fifos.get_small_entry(offset);
                entry.incr_recency();
            }
            LocalOffset::Main(offset) => {
                let entry = self.fifos.get_main_entry(offset);
                entry.incr_recency();
            }
        }
    }

    /// Push to main queue. Evict or reinsert based on recency.
    #[must_use]
    fn push_to_main_queue(&mut self, entry: Entry<K, V>) -> LocalOffset {
        // Try fast path
        let entry = match self.fifos.main.try_push(entry) {
            Ok(off) => return LocalOffset::Main(off as u32),
            Err(entry) => entry,
        };

        // Reinsert while entries with recency > 0 exist
        while self.fifos.main.reinsert_unchecked_if(|e| {
            if e.recency() > 0 {
                e.decr_recency();
                true
            } else {
                false
            }
        }) {}

        // We need to remove from hashtable BEFORE overwriting the slot,
        // because remove_from_hashtable uses key_eq which reads the current
        // slot contents. If we remove after overwriting_push, the slot
        // already contains the new key, so key_eq fails and the entry
        // becomes a zombie (never removed).
        let evict_position = self.fifos.main.write_position();
        let evicted_key = self.fifos.main.get_absolute_unchecked(evict_position).key;
        self.remove_from_hashtable(&evicted_key);

        // Now safe to overwrite the slot
        let position = self.fifos.main.overwriting_push(entry);

        LocalOffset::Main(position as u32)
    }

    /// Push to small queue, handling eviction to main/ghost and updating hashtable.
    #[must_use]
    fn push_to_small_queue(&mut self, entry: Entry<K, V>) -> LocalOffset {
        // Try fast path
        let entry = match self.fifos.small.try_push(entry) {
            Ok(off) => return LocalOffset::Small(off as u32),
            Err(entry) => entry,
        };

        // We need to read the oldest entry and update the hashtable BEFORE
        // overwriting the slot. This is because update_hashtable use key_eq
        // which reads the current slot contents. If we do this after overwriting_push,
        // the slot contains the new key, so key_eq fails and we fail to update the hashtable.
        let oldest_offset = self.fifos.small.write_position();
        let oldest_entry = self.fifos.small.get_absolute_unchecked(oldest_offset);
        let oldest_key = oldest_entry.key;

        let new_offset = if oldest_entry.recency() > 0 {
            // Promote to main queue
            self.push_to_main_queue(oldest_entry.clone())
        } else {
            // Demote key to ghost queue
            self.push_to_ghost_queue(oldest_key)
        };

        // Update the hashtable to now find the moved entry at the main/ghost queue.
        // SAFETY: Entry is currently at two places, but:
        // 1. We have removed the bucket pointing to the overwritten position in main/ghost.
        // 2. So key_eq will only succeed with the one in small queue.
        self.update_hashtable(&oldest_key, new_offset);

        // Now safe to overwrite the slot
        let offset = self.fifos.small.overwriting_push(entry);

        LocalOffset::Small(offset as u32)
    }

    fn push_to_ghost_queue(&mut self, key: K) -> LocalOffset {
        // Try fast path
        let key = match self.fifos.ghost.try_push(key) {
            Ok(offset) => return LocalOffset::Ghost(offset as u32),
            Err(key) => key,
        };

        // We need to remove from hashtable BEFORE overwriting the slot,
        // because remove_from_hashtable uses key_eq which reads the current
        // slot contents. If we remove after pop_push_unchecked, the slot
        // already contains the new key, so key_eq fails and the entry
        // becomes a zombie (never removed).
        let evict_offset = self.fifos.ghost.write_position();
        let evicted_key = *self.fifos.get_ghost_key(evict_offset as u32); // K: Copy
        self.remove_from_hashtable(&evicted_key);

        // Now safe to overwrite the slot
        let offset = self.fifos.ghost.overwriting_push(key);

        LocalOffset::Ghost(offset as u32)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::thread;
    use std::time::Duration;

    use rand::Rng;

    use super::*;

    #[test]
    fn basic_insert_get() {
        let cache = Cache::<u64, String>::new(100);

        cache.insert(1, "hello".to_string());
        thread::sleep(Duration::from_millis(10));

        let got = cache.get(&1);
        assert_eq!(got, Some("hello".to_string()));
    }

    #[test]
    fn concurrent_reads() {
        let cache = Cache::<u64, String>::new(200);

        for i in 0..20u64 {
            cache.insert(i, format!("val_{i}"));
        }

        thread::sleep(Duration::from_millis(50));

        let arc = Arc::new(cache);
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let c = Arc::clone(&arc);
                thread::spawn(move || {
                    for _ in 0..100 {
                        for k in 0..20u64 {
                            let _ = c.get(&k);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn multi_threaded_inserts() {
        let cache = Arc::new(Cache::<u64, u64>::new(5000));

        const THREADS: usize = 4;
        const PER: u64 = 1000;

        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let c = Arc::clone(&cache);
                thread::spawn(move || {
                    let start = (t as u64) * PER;
                    for i in start..(start + PER) {
                        c.insert(i, i * 2);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        thread::sleep(Duration::from_millis(100));

        // At least some entries should be present
        let mut found = 0usize;
        for i in 0..(THREADS as u64 * PER) {
            if cache.get(&i).is_some() {
                found += 1;
            }
        }
        assert!(found > 0);
    }

    #[test]
    fn len_is_empty() {
        let cache = Cache::<u64, String>::new(100);
        assert!(cache.is_empty());
        cache.insert(42, "v".to_string());
        thread::sleep(Duration::from_millis(10));
        assert!(!cache.is_empty());
    }

    #[test]
    fn high_contention_inserts() {
        let cache = Arc::new(Cache::<u64, u64>::new(2000));
        let counter = Arc::new(AtomicUsize::new(0));

        const THREADS: usize = 8;
        const INSERTS: usize = 500;

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let c = Arc::clone(&cache);
                let ctr = Arc::clone(&counter);
                thread::spawn(move || {
                    for _ in 0..INSERTS {
                        let k = ctr.fetch_add(1, Ordering::Relaxed) as u64;
                        c.insert(k, k);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        thread::sleep(Duration::from_millis(100));
        assert!(!cache.is_empty());
    }

    // --- New fuzz test using repo RNG and a parallel seen map -----------------

    /// Heavy fuzz test that uses the repository RNG utilities and a parallel map
    /// of seen values per key to ensure the cache never returns a value that
    /// was not previously inserted for that key.
    #[test]
    fn fuzz_never_returns_unseen_value() {
        const CAPACITY: usize = 1024;
        let cache = Arc::new(Cache::<u64, u64>::new(CAPACITY));

        const KEY_SPACE: u64 = 10240;
        const WRITERS: usize = 6;
        const READERS: usize = 6;
        const OPS_PER_WRITER: usize = 200_000;

        // Pre-generate a hashmap of random keys -> sets of allowed values.
        // This map is generated once up-front and then shared (read-only) by all
        // threads. Any value returned by the cache must belong to the set for
        // that key (or the key may not be present at all).
        let mut initial: HashMap<u64, Vec<u64>> = HashMap::new();
        for k in 0..KEY_SPACE {
            // Give each key between 1 and 8 candidate values.
            let count = (rand::random::<u8>() as usize % 8) + 1;
            let mut set = HashSet::new();
            while set.len() < count {
                set.insert(rand::random::<u64>());
            }
            initial.insert(k, set.into_iter().collect());
        }
        let initial = Arc::new(initial);

        let mut handles = Vec::with_capacity(WRITERS + READERS);

        // Writers: randomly pick a key and a value from the pre-generated map and insert.
        for _ in 0..WRITERS {
            let c = Arc::clone(&cache);
            let initial = Arc::clone(&initial);
            handles.push(thread::spawn(move || {
                let mut rnd = rand::rng();
                for _ in 0..OPS_PER_WRITER {
                    let key = rnd.random_range(0..KEY_SPACE);
                    let vec = &initial[&key];
                    let value = vec[rnd.random_range(0..vec.len())];
                    c.insert(key, value);
                    // occasional yield to increase interleaving
                    if (rnd.random::<u32>() & 0x3ff) == 0 {
                        thread::yield_now();
                    }
                }
            }));
        }

        // Readers: randomly probe keys and verify any returned value belongs to the pre-generated set.
        for _ in 0..READERS {
            let c = Arc::clone(&cache);
            let initial = Arc::clone(&initial);
            handles.push(thread::spawn(move || {
                let mut rnd = rand::rng();
                // number of reads is proportional to total writes
                let reads = (OPS_PER_WRITER * WRITERS) / (READERS * 2).max(1);
                for _ in 0..reads {
                    let key = rnd.random_range(0..KEY_SPACE);
                    if let Some(v) = c.get(&key) {
                        let vec = &initial[&key];
                        // ensure returned value is one of the pre-generated values for this key
                        assert!(
                            vec.contains(&v),
                            "cache returned unseen value {v} for key {key}",
                        );
                    }
                    // occasionally yield
                    if (rnd.random::<u8>() & 0x1f) == 0 {
                        thread::yield_now();
                    }
                }
            }));
        }

        // Wait for all threads to finish
        for h in handles {
            h.join().unwrap();
        }

        // Allow the writer/disruptor to flush events
        thread::sleep(Duration::from_millis(200));

        // Final verification: any value returned by the cache must belong to the pre-generated set
        for k in 0..KEY_SPACE {
            if let Some(v) = cache.get(&k) {
                let vec = &initial[&k];
                assert!(
                    vec.contains(&v),
                    "final check: returned {v} for key {k} which wasn't in the initial set",
                );
            }
        }
    }
}
