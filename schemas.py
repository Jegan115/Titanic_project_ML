from typing import Optional

from pydantic import BaseModel

class Passenger(BaseModel):

    pclass: int
    sex: str

    age:Optional[float] = None
    fare:Optional[float] = None