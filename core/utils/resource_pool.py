from typing import TypeVar, Generic, List, Optional, Callable, Any
from threading import Lock, Condition
import time
import weakref
from core.utils.logging import Logger

T = TypeVar('T')

class ResourcePool(Generic[T]):
    """Enhanced resource pool with camera frame management"""

    def __init__(self, max_size: int = 10, resource_type: str = "unknown",
                 factory: Optional[Callable[[], T]] = None,
                 cleanup: Optional[Callable[[T], None]] = None,
                 max_idle_time: float = 300.0):  # 5 minutes

        self.max_size = max_size
        self.resource_type = resource_type
        self.factory = factory
        self.cleanup = cleanup
        self.max_idle_time = max_idle_time

        self.pool: List[tuple[T, float]] = []  # (resource, last_used_time)
        self.lock = Lock()
        self.condition = Condition(self.lock)
        self.logger = Logger()

        # Statistics
        self.stats = {
            'created': 0,
            'reused': 0,
            'discarded': 0,
            'cleaned_up': 0
        }

        # Weak references to track all created resources
        self._all_resources = weakref.WeakSet()

    def get_resource(self, timeout: Optional[float] = None) -> Optional[T]:
        """Get a resource from the pool with optional timeout"""
        with self.condition:
            start_time = time.time()

            while True:
                # Clean up expired resources first
                self._cleanup_expired_resources()

                # Try to get an existing resource
                if self.pool:
                    resource, _ = self.pool.pop()
                    self.stats['reused'] += 1
                    self.logger.debug(f"Retrieved {self.resource_type} from pool (reused)")
                    return resource

                # Create new resource if factory is available
                if self.factory and len(self._all_resources) < self.max_size:
                    try:
                        resource = self.factory()
                        self._all_resources.add(resource)
                        self.stats['created'] += 1
                        self.logger.debug(f"Created new {self.resource_type}")
                        return resource
                    except Exception as e:
                        self.logger.error(f"Failed to create {self.resource_type}: {e}")
                        return None

                # Wait for resource if timeout specified
                if timeout is None:
                    self.logger.debug(f"No {self.resource_type} available in pool")
                    return None

                elapsed = time.time() - start_time
                remaining = timeout - elapsed

                if remaining <= 0:
                    self.logger.warning(f"Timeout waiting for {self.resource_type}")
                    return None

                self.condition.wait(remaining)

    def return_resource(self, resource: T, force_cleanup: bool = False):
        """Return a resource to the pool"""
        with self.condition:
            if force_cleanup or len(self.pool) >= self.max_size:
                # Pool is full or cleanup forced, destroy resource
                self._destroy_resource(resource)
                self.stats['discarded'] += 1
                self.logger.debug(f"Discarded {self.resource_type} (pool full or forced)")
            else:
                # Add to pool with timestamp
                current_time = time.time()
                self.pool.append((resource, current_time))
                self.logger.debug(f"Returned {self.resource_type} to pool")

            # Notify waiting threads
            self.condition.notify()

    def _cleanup_expired_resources(self):
        """Clean up resources that have been idle too long"""
        if self.max_idle_time <= 0:
            return

        current_time = time.time()
        expired_indices = []

        for i, (resource, last_used) in enumerate(self.pool):
            if current_time - last_used > self.max_idle_time:
                expired_indices.append(i)

        # Remove expired resources (reverse order to maintain indices)
        for i in reversed(expired_indices):
            resource, _ = self.pool.pop(i)
            self._destroy_resource(resource)
            self.stats['cleaned_up'] += 1
            self.logger.debug(f"Cleaned up expired {self.resource_type}")

    def _destroy_resource(self, resource: T):
        """Safely destroy a resource"""
        try:
            if self.cleanup:
                self.cleanup(resource)
            # Remove from weak set if possible
            self._all_resources.discard(resource)
        except Exception as e:
            self.logger.error(f"Error destroying {self.resource_type}: {e}")

    def clear(self):
        """Clear all resources from the pool"""
        with self.lock:
            for resource, _ in self.pool:
                self._destroy_resource(resource)
            self.pool.clear()
            self.logger.info(f"Cleared {self.resource_type} pool")

    def size(self) -> int:
        """Get current pool size"""
        with self.lock:
            return len(self.pool)

    def total_resources(self) -> int:
        """Get total number of created resources"""
        return len(self._all_resources)

    def is_full(self) -> bool:
        """Check if pool is full"""
        return self.size() >= self.max_size

    def get_stats(self) -> dict:
        """Get pool statistics"""
        with self.lock:
            return {
                **self.stats.copy(),
                'pool_size': len(self.pool),
                'total_resources': len(self._all_resources),
                'pool_utilization': len(self.pool) / self.max_size if self.max_size > 0 else 0
            }

    def cleanup_all(self):
        """Cleanup all resources including those outside the pool"""
        with self.lock:
            # Clear the pool first
            self.clear()

            # Try to cleanup all tracked resources
            resources_to_cleanup = list(self._all_resources)
            for resource in resources_to_cleanup:
                self._destroy_resource(resource)

            self.logger.info(f"Cleaned up all {self.resource_type} resources")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup_all()

    def __del__(self):
        """Destructor cleanup"""
        try:
            self.cleanup_all()
        except:
            pass  # Ignore errors during destruction


# Specialized frame pool for camera service
class FramePool(ResourcePool):
    """Specialized resource pool for camera frames"""

    def __init__(self, max_size: int = 5):
        import numpy as np

        def create_frame():
            # Create a placeholder frame
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        def cleanup_frame(frame):
            # Clear frame data
            if hasattr(frame, 'fill'):
                frame.fill(0)

        super().__init__(
            max_size=max_size,
            resource_type="camera_frame",
            factory=create_frame,
            cleanup=cleanup_frame,
            max_idle_time=60.0  # 1 minute for frames
        )
