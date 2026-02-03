use std::sync::Arc;

use common::budget::ResourceBudget;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::save_on_disk::SaveOnDisk;
use segment::data_types::vectors::VectorStructInternal;
use segment::types::{PayloadFieldSchema, PayloadSchemaType};
use shard::operations::CollectionUpdateOperations;
use shard::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted,
};
use tempfile::Builder;
use tokio::runtime::Handle;
use tokio::sync::RwLock;

use crate::operations::shared_storage_config::SharedStorageConfig;
use crate::shards::local_shard::LocalShard;
use crate::shards::shard_trait::ShardOperation;
use crate::tests::fixtures::*;
use crate::update_workers::applied_seq::AppliedSeqHandler;

#[tokio::test(flavor = "multi_thread")]
async fn test_delete_from_indexed_payload() {
    //  Init the logger
    let _ = env_logger::builder().is_test(true).try_init();
    let collection_dir = Builder::new().prefix("test_collection").tempdir().unwrap();

    let config = create_collection_config();

    let collection_name = "test".to_string();

    let current_runtime: Handle = Handle::current();

    let payload_index_schema_dir = Builder::new().prefix("qdrant-test").tempdir().unwrap();
    let payload_index_schema_file = payload_index_schema_dir.path().join("payload-schema.json");
    let payload_index_schema =
        Arc::new(SaveOnDisk::load_or_init_default(payload_index_schema_file).unwrap());

    let shard = LocalShard::build(
        0,
        collection_name.clone(),
        collection_dir.path(),
        Arc::new(RwLock::new(config.clone())),
        Arc::new(Default::default()),
        payload_index_schema.clone(),
        current_runtime.clone(),
        current_runtime.clone(),
        ResourceBudget::default(),
        config.optimizer_config.clone(),
    )
    .await
    .unwrap();

    let upsert_ops = upsert_operation();

    let hw_acc = HwMeasurementAcc::new();

    shard
        .update(upsert_ops.into(), true, None, hw_acc.clone())
        .await
        .unwrap();

    let index_op = create_payload_index_operation();

    payload_index_schema
        .write(|schema| {
            schema.schema.insert(
                "location".parse().unwrap(),
                PayloadFieldSchema::FieldType(PayloadSchemaType::Geo),
            );
        })
        .unwrap();
    shard
        .update(index_op.into(), true, None, hw_acc.clone())
        .await
        .unwrap();

    let delete_point_op = delete_point_operation(4);
    shard
        .update(delete_point_op.into(), true, None, hw_acc.clone())
        .await
        .unwrap();

    let info = shard.info().await.unwrap();
    eprintln!("info = {:#?}", info.payload_schema);
    let number_of_indexed_points = info
        .payload_schema
        .get(&"location".parse().unwrap())
        .unwrap()
        .points;

    shard.stop_gracefully().await;

    let shard = LocalShard::load(
        0,
        collection_name.clone(),
        collection_dir.path(),
        Arc::new(RwLock::new(config.clone())),
        config.optimizer_config.clone(),
        Arc::new(Default::default()),
        payload_index_schema.clone(),
        true,
        current_runtime.clone(),
        current_runtime.clone(),
        ResourceBudget::default(),
    )
    .await
    .unwrap();

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    eprintln!("dropping point 5");
    let delete_point_op = delete_point_operation(5);
    shard
        .update(delete_point_op.into(), true, None, hw_acc.clone())
        .await
        .unwrap();

    shard.stop_gracefully().await;

    let shard = LocalShard::load(
        0,
        collection_name,
        collection_dir.path(),
        Arc::new(RwLock::new(config.clone())),
        config.optimizer_config.clone(),
        Arc::new(Default::default()),
        payload_index_schema,
        true,
        current_runtime.clone(),
        current_runtime,
        ResourceBudget::default(),
    )
    .await
    .unwrap();

    let info = shard.info().await.unwrap();
    eprintln!("info = {:#?}", info.payload_schema);

    let number_of_indexed_points_after_load = info
        .payload_schema
        .get(&"location".parse().unwrap())
        .unwrap()
        .points;

    assert_eq!(number_of_indexed_points, 4);
    assert_eq!(number_of_indexed_points_after_load, 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_partial_flush_recovery() {
    //  Init the logger
    let _ = env_logger::builder().is_test(true).try_init();
    let collection_dir = Builder::new().prefix("test_collection").tempdir().unwrap();

    let config = create_collection_config();

    let collection_name = "test".to_string();

    let current_runtime: Handle = Handle::current();

    let payload_index_schema_dir = Builder::new().prefix("qdrant-test").tempdir().unwrap();
    let payload_index_schema_file = payload_index_schema_dir.path().join("payload-schema.json");
    let payload_index_schema =
        Arc::new(SaveOnDisk::load_or_init_default(payload_index_schema_file).unwrap());

    let shard = LocalShard::build(
        0,
        collection_name.clone(),
        collection_dir.path(),
        Arc::new(RwLock::new(config.clone())),
        Arc::new(Default::default()),
        payload_index_schema.clone(),
        current_runtime.clone(),
        current_runtime.clone(),
        ResourceBudget::default(),
        config.optimizer_config.clone(),
    )
    .await
    .unwrap();

    let upsert_ops = upsert_operation();

    let hw_acc = HwMeasurementAcc::new();

    shard
        .update(upsert_ops.into(), true, None, hw_acc.clone())
        .await
        .unwrap();

    let index_op = create_payload_index_operation();

    payload_index_schema
        .write(|schema| {
            schema.schema.insert(
                "location".parse().unwrap(),
                PayloadFieldSchema::FieldType(PayloadSchemaType::Geo),
            );
        })
        .unwrap();

    shard
        .update(index_op.into(), true, None, hw_acc.clone())
        .await
        .unwrap();

    shard.stop_flush_worker().await;

    shard.full_flush();

    let delete_point_op = delete_point_operation(4);
    shard
        .update(delete_point_op.into(), true, None, hw_acc.clone())
        .await
        .unwrap();

    // This only flushed id-tracker-mapping, but not the storage change
    shard.partial_flush();

    shard.stop_gracefully().await;

    let shard = LocalShard::load(
        0,
        collection_name,
        collection_dir.path(),
        Arc::new(RwLock::new(config.clone())),
        config.optimizer_config.clone(),
        Arc::new(Default::default()),
        payload_index_schema,
        true,
        current_runtime.clone(),
        current_runtime,
        ResourceBudget::default(),
    )
    .await
    .unwrap();

    let info = shard.info().await.unwrap();
    eprintln!("info = {:#?}", info.payload_schema);

    let number_of_indexed_points_after_load = info
        .payload_schema
        .get(&"location".parse().unwrap())
        .unwrap()
        .points;

    assert_eq!(number_of_indexed_points_after_load, 4);
}

/// Test that verifies the WAL recovery process correctly loads pending updates into the update queue.
#[tokio::test(flavor = "multi_thread")]
async fn test_wal_replay_loads_pending_to_queue() {
    let _ = env_logger::builder().is_test(true).try_init();
    let collection_dir = Builder::new().prefix("test_collection").tempdir().unwrap();

    let config = create_collection_config();

    let collection_name = "test".to_string();

    let current_runtime: Handle = Handle::current();

    let payload_index_schema_dir = Builder::new().prefix("qdrant-test").tempdir().unwrap();
    let payload_index_schema_file = payload_index_schema_dir.path().join("payload-schema.json");
    let payload_index_schema =
        Arc::new(SaveOnDisk::load_or_init_default(payload_index_schema_file).unwrap());

    let shared_storage_config = Arc::new(SharedStorageConfig {
        update_queue_size: 10_000,
        ..Default::default()
    });

    // We need WAL length > applied_seq + 65 to trigger the queue loading path
    let total_ops = 500u64;

    let shard = LocalShard::build(
        0,
        collection_name.clone(),
        collection_dir.path(),
        Arc::new(RwLock::new(config.clone())),
        shared_storage_config.clone(),
        payload_index_schema.clone(),
        current_runtime.clone(),
        current_runtime.clone(),
        ResourceBudget::default(),
        config.optimizer_config.clone(),
    )
    .await
    .unwrap();

    shard.stop_flush_worker().await;

    let hw_acc = HwMeasurementAcc::new();

    // Insert all operations
    for i in 0..total_ops {
        let point = PointStructPersisted {
            id: i.into(),
            vector: VectorStructInternal::from(vec![1.0, 2.0, 3.0, 4.0]).into(),
            payload: None,
        };
        let op = CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
            PointInsertOperationsInternal::PointsList(vec![point]),
        ));
        shard
            .update(op.into(), true, None, hw_acc.clone())
            .await
            .unwrap();
    }

    // Stop the shard without flush to preserve WAL.
    shard.stop_gracefully().await;

    // Use AppliedSeqHandler to read and manipulate the applied_seq file.
    // This simulates a scenario where applied_seq is lower than the actual WAL length.
    // We need to ensure: WAL first_index <= applied_seq < WAL last_index - 65
    // The WAL might be truncated, so it's important to be careful with the replaced value.
    let applied_seq_handler = AppliedSeqHandler::load_or_init(collection_dir.path(), total_ops);
    eprintln!("Applied seq path: {:?}", applied_seq_handler.path());

    // Read the current applied_seq value
    let current_applied_seq = applied_seq_handler.op_num().unwrap_or(0);
    eprintln!("Current applied_seq: {current_applied_seq}");

    // Calculate the target low value:
    // - upper_bound = (total_ops - 100) + 64 = total_ops - 36
    // - At least 36 entries should go to update queue.
    // It's ok if they are applied already, segment will skip them.
    // We only need to ensure they are in the update queue.
    let low_applied_seq = total_ops.saturating_sub(100);

    // Only modify if the current value is too large
    if current_applied_seq > low_applied_seq {
        applied_seq_handler
            .force_set_and_persist(low_applied_seq)
            .unwrap();
        eprintln!(
            "Reduced applied_seq from {current_applied_seq} to {low_applied_seq}, total_ops: {total_ops}"
        );
    } else {
        eprintln!(
            "Applied_seq {current_applied_seq} is already <= target {low_applied_seq}, total_ops: {total_ops}"
        );
    }

    // Reload the shard
    let shard = LocalShard::load(
        0,
        collection_name,
        collection_dir.path(),
        Arc::new(RwLock::new(config.clone())),
        config.optimizer_config.clone(),
        shared_storage_config,
        payload_index_schema,
        false,
        current_runtime.clone(),
        current_runtime.clone(),
        ResourceBudget::default(),
    )
    .await
    .unwrap();

    // Check update queue info immediately after load.
    let post_load_info = shard.local_update_queue_info();
    eprintln!("Post-load update queue info: {post_load_info:?}");

    // The applied_seq should be the value we set
    assert!(
        post_load_info.op_num.is_some(),
        "applied_seq should be tracked"
    );

    // Length should be not zero, as there should be pending ops loaded into the queue.
    assert!(
        post_load_info.length > 0,
        "update queue should have pending operations after WAL replay"
    );

    let loaded_applied_seq = post_load_info.op_num.unwrap();
    assert_eq!(
        loaded_applied_seq, low_applied_seq as usize,
        "applied_seq should match the value we set in the file"
    );

    // Wait for update worker to process all queued operations with a timeout
    let timeout = std::time::Duration::from_secs(2);
    let start = std::time::Instant::now();
    let poll_interval = std::time::Duration::from_millis(10);

    while shard.local_update_queue_info().length > 0 {
        assert!(
            start.elapsed() <= timeout,
            "Timeout waiting for update queue to empty"
        );
        tokio::time::sleep(poll_interval).await;
    }

    // Verify all points are present after processing
    let info = shard.info().await.unwrap();
    let points_count = info.points_count.unwrap_or(0);
    assert_eq!(
        points_count, total_ops as usize,
        "All {total_ops} points should be present after WAL replay and update queue processing",
    );

    shard.stop_gracefully().await;
}
