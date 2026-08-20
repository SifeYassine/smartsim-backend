from typing import Optional, Union
from sqlalchemy.types import TypeDecorator, LargeBinary
import numpy as np

class Vector(TypeDecorator):
  impl = LargeBinary
  cache_ok = True

  def process_bind_param(self, value: Optional[Union[list[float], np.ndarray]], dialect) -> Optional[bytes]:
    if value is None:
      return None
    return np.array(value, dtype=np.float32).tobytes()

  def process_result_value(self, value: Optional[bytes], dialect) -> Optional[list[float]]:
    if value is None:
      return None
    return np.frombuffer(value, dtype=np.float32).tolist()