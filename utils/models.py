from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Literal, Dict


class Txn_data(BaseModel):
    transaction_amount: float	
    transaction_type: Literal['Online', 'POS', 'ATM', 'Transfer']	
    device_type: Literal['Mobile', 'ATM Machine', 'POS Terminal', 'Web']
    location: Literal['Abuja', 'Lagos', 'Ibadan', 'Kano', 'Port Harcourt']	
    is_foreign_transaction: int = Field(..., ge=0, le=1, description="Binary class: 0 or 1")
    is_high_risk_country: int = Field(..., ge=0, le=1, description="Binary class: 0 or 1")
    previous_fraud_flag: int = Field(..., ge=0, le=1, description="Binary class: 0 or 1")	
    transaction_time: datetime
    risk_score: float  
    
    @field_validator("transaction_time", mode="before")
    @classmethod
    def validate_transaction_time(cls, v):
        if isinstance(v, datetime):
            return v
        try:
            return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError("transaction_time must be in 'YYYY:MM:DD HH:MM:SS' format")
        
        
class PredictionResponse(BaseModel):
    prediction: str
    explanation: Dict[str, float]