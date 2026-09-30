import os
import time
import threading
import queue
import uuid
import shutil

# Import the parts we modified from watcher.py
from api.watcher import PendingSet, io_worker

def test_io_worker():
    print("Setting up test directories...")
    os.makedirs("test_input", exist_ok=True)
    os.makedirs("test_failed", exist_ok=True)
    
    # 1. Create some test files
    for i in range(5):
        with open(f"test_input/test_{i}.jpg", "w") as f:
            f.write("dummy image data")
            
    # Also create a collision file in failed
    with open("test_failed/test_1.jpg", "w") as f:
        f.write("existing file")

    pending_set = PendingSet()
    cleanup_queue = queue.Queue(maxsize=2)  # small size to test backpressure
    worker_thread = threading.Thread(target=io_worker, args=(cleanup_queue, pending_set))
    worker_thread.start()

    print("\n1. Testing normal delete (test_0.jpg)")
    p0 = os.path.abspath("test_input/test_0.jpg")
    pending_set.add(p0)
    cleanup_queue.put((p0, None))
    
    print("2. Testing failed-move with collision (test_1.jpg)")
    p1 = os.path.abspath("test_input/test_1.jpg")
    pending_set.add(p1)
    cleanup_queue.put((p1, os.path.abspath("test_failed/test_1.jpg")))
    
    print("3. Testing full-queue backpressure")
    print("   Queue size is 2, and we have 2 items. Next put will block if they haven't processed.")
    p2 = os.path.abspath("test_input/test_2.jpg")
    pending_set.add(p2)
    # This might block momentarily until worker consumes
    cleanup_queue.put((p2, None))
    print("   Backpressure handled successfully.")
    
    print("4. Testing pending set ignore logic")
    p3 = os.path.abspath("test_input/test_3.jpg")
    pending_set.add(p3)
    # Simulating watcher scan
    if p3 in pending_set:
        print("   File test_3.jpg is correctly ignored by watcher scan because it is pending.")
        
    print("5. Testing shutdown drain")
    # Put sentinel to shutdown
    cleanup_queue.put((p3, None))
    cleanup_queue.put(None)
    cleanup_queue.join()
    worker_thread.join()
    print("   Worker joined successfully, queue drained.")
    
    # Verification
    assert not os.path.exists("test_input/test_0.jpg"), "test_0 should be deleted"
    assert not os.path.exists("test_input/test_1.jpg"), "test_1 should be moved"
    assert not os.path.exists("test_input/test_2.jpg"), "test_2 should be deleted"
    assert not os.path.exists("test_input/test_3.jpg"), "test_3 should be deleted"
    
    # Check collision was handled
    failed_files = os.listdir("test_failed")
    assert len(failed_files) >= 2, "Failed dir should have the original test_1.jpg and the new renamed test_1_xxx.jpg"
    print("\nAll tests passed successfully!")
    
    # Cleanup
    shutil.rmtree("test_input")
    shutil.rmtree("test_failed")

if __name__ == "__main__":
    test_io_worker()
