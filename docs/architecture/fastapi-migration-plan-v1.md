# Flask to FastAPI Migration Plan: Domain-Oriented Architecture

This document outlines the migration strategy for converting biocentral_server from Flask to FastAPI, focusing on core protein analysis workflows (embeddings → predictions) and transitioning to a domain-oriented architecture.

## Migration Overview and Strategic Focus

### Primary Goals
1. **Core Workflow Focus**: Prioritize embeddings and embedding-based predictions (TMbed, VespaG, etc.)
2. **Domain-Oriented Architecture**: Restructure around business domains rather than technical layers
3. **Framework Migration**: Convert from Flask + Blueprints to FastAPI + Routers
4. **Logic Decoupling**: Separate business logic from server framework dependencies
5. **Type Safety**: Implement Pydantic models for domain-specific validation

### Core Protein Analysis Workflow
```
Protein Sequences → Embeddings → Predictions (TMbed/VespaG/etc.) → Results
```

This represents the primary value chain that drives most system usage and should be the migration priority.

## Domain-Oriented Architecture Design

### Current Technical Architecture (Flask-based)
- **9 technical service modules**: embeddings, predictions, proteins, ppi, etc.
- **Cross-cutting concerns**: Mixed throughout service modules
- **Framework coupling**: Business logic tightly coupled to Flask patterns

### Target Domain Architecture (FastAPI-based)

#### Core Domains (Priority 1)
1. **Protein Embeddings Domain**
   - Embedding generation (ProtT5, ESM, etc.)
   - Embedding storage and retrieval
   - Embedding comparison and similarity

2. **Protein Predictions Domain**
   - Structure predictions (TMbed for membrane topology)
   - Function predictions (VespaG for variant effects)
   - Property predictions (disorder, localization, etc.)

#### Supporting Domains (Priority 2)
3. **Protein Analysis Domain**
   - Sequence analysis and validation
   - Similarity calculations
   - Statistical analysis

4. **Model Training Domain**
   - Biotrainer workflow management
   - Model evaluation and validation
   - Training pipeline orchestration

#### Infrastructure Domains (Priority 3)
5. **Task Management Domain**
   - Background job orchestration
   - Progress tracking and notifications
   - Resource allocation

6. **Data Management Domain**
   - File storage and retrieval
   - Database operations
   - Data validation and transformation

## Core Workflow Migration Strategy

### Phase 1: Protein Embeddings Domain (Weeks 1-3)

#### 1.1 Domain-Oriented Project Structure
```
biocentral_server/
├── domains/
│   ├── protein_embeddings/        # CORE DOMAIN
│   │   ├── __init__.py
│   │   ├── models/                # Domain-specific Pydantic models
│   │   │   ├── __init__.py
│   │   │   ├── embedding_request.py
│   │   │   ├── embedding_response.py
│   │   │   └── embedding_entities.py
│   │   ├── services/              # Domain business logic
│   │   │   ├── __init__.py
│   │   │   ├── embedding_service.py
│   │   │   ├── embedding_storage_service.py
│   │   │   └── embedding_computation_service.py
│   │   ├── repositories/          # Data access layer
│   │   │   ├── __init__.py
│   │   │   ├── embedding_repository.py
│   │   │   └── sequence_repository.py
│   │   └── api/                   # Domain API endpoints
│   │       ├── __init__.py
│   │       └── embedding_router.py
│   │
│   ├── protein_predictions/       # CORE DOMAIN
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── prediction_request.py
│   │   │   ├── prediction_response.py
│   │   │   ├── tmbed_models.py
│   │   │   └── vespag_models.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── prediction_service.py
│   │   │   ├── tmbed_service.py     # Membrane topology
│   │   │   ├── vespag_service.py    # Variant effects
│   │   │   ├── disorder_service.py  # Protein disorder
│   │   │   └── localization_service.py
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   └── model_repository.py
│   │   └── api/
│   │       ├── __init__.py
│   │       └── prediction_router.py
│   │
│   └── shared/                    # Cross-domain utilities
│       ├── __init__.py
│       ├── task_management/       # Background jobs
│       ├── data_access/          # Database connections
│       └── validation/           # Common validators
│
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app
│   ├── dependencies.py           # Cross-domain dependencies
│   └── middleware.py            # CORS, auth, etc.
│
└── infrastructure/              # Technical concerns
    ├── __init__.py
    ├── database/
    ├── storage/
    └── monitoring/
```

#### 1.2 Embedding Domain Models
```python
# domains/protein_embeddings/models/embedding_request.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from enum import Enum

class SupportedEmbeddingModel(str, Enum):
    PROT_T5 = "ProtT5"
    ESM_2 = "ESM-2"
    ANKH = "Ankh"

class EmbeddingRequest(BaseModel):
    sequences: List[str] = Field(
        ...,
        description="Protein sequences to embed",
        min_items=1,
        max_items=1000
    )
    model_name: SupportedEmbeddingModel = Field(
        SupportedEmbeddingModel.PROT_T5,
        description="Embedding model to use"
    )
    batch_size: Optional[int] = Field(
        32,
        description="Batch size for processing",
        ge=1,
        le=128
    )
    return_raw_embeddings: bool = Field(
        False,
        description="Return full embeddings or just store them"
    )

    @validator('sequences')
    def validate_sequences(cls, v):
        for seq in v:
            if len(seq) > 5000:
                raise ValueError(f'Sequence too long (max 5000 characters): {len(seq)}')
            if not seq.replace(' ', '').isalpha():
                raise ValueError('Sequence contains invalid characters')
        return v

class BatchEmbeddingRequest(BaseModel):
    """For processing large batches of sequences"""
    sequence_ids: List[str] = Field(..., description="Sequence identifiers")
    sequences: List[str] = Field(..., description="Protein sequences")
    model_name: SupportedEmbeddingModel = Field(SupportedEmbeddingModel.PROT_T5)
    priority: Literal["high", "normal", "low"] = Field("normal")
```

#### 1.3 Embedding Domain Service
```python
# domains/protein_embeddings/services/embedding_service.py
from typing import List, Dict, Optional, Union
import numpy as np
from ..models.embedding_request import EmbeddingRequest, BatchEmbeddingRequest
from ..models.embedding_response import EmbeddingResponse, EmbeddingTaskResponse
from ..repositories.embedding_repository import EmbeddingRepository
from ...shared.task_management.task_manager import TaskManager

class ProteinEmbeddingService:
    """Core domain service for protein embeddings - decoupled from web framework"""

    def __init__(
        self,
        embedding_repository: EmbeddingRepository,
        task_manager: TaskManager
    ):
        self.embedding_repo = embedding_repository
        self.task_manager = task_manager

    async def process_embedding_request(
        self,
        request: EmbeddingRequest,
        user_id: str
    ) -> Union[EmbeddingResponse, EmbeddingTaskResponse]:
        """Main embedding workflow orchestration"""

        # Check which embeddings already exist
        missing_sequences = await self.embedding_repo.get_missing_embeddings(
            request.sequences,
            request.model_name
        )

        if not missing_sequences:
            # All embeddings exist, return immediately
            if request.return_raw_embeddings:
                embeddings = await self.embedding_repo.get_embeddings(
                    request.sequences,
                    request.model_name
                )
                return EmbeddingResponse(
                    embeddings=embeddings,
                    model_used=request.model_name,
                    cached=True
                )
            else:
                return EmbeddingResponse(
                    message="All embeddings already exist in database",
                    model_used=request.model_name,
                    cached=True
                )

        # Start background task for missing embeddings
        task_id = await self.task_manager.enqueue_embedding_task(
            sequences=missing_sequences,
            model_name=request.model_name,
            batch_size=request.batch_size,
            user_id=user_id,
            return_embeddings=request.return_raw_embeddings
        )

        return EmbeddingTaskResponse(
            task_id=task_id,
            status="pending",
            message=f"Computing {len(missing_sequences)} missing embeddings",
            total_sequences=len(request.sequences),
            missing_sequences=len(missing_sequences)
        )

    async def get_embeddings_for_prediction(
        self,
        sequences: List[str],
        model_name: str = "ProtT5"
    ) -> Dict[str, np.ndarray]:
        """Specialized method for prediction workflows"""
        return await self.embedding_repo.get_embeddings_as_arrays(
            sequences,
            model_name
        )
```

### Phase 2: Protein Predictions Domain (Weeks 4-6)

#### 2.1 Prediction Domain Models
```python
# domains/protein_predictions/models/prediction_request.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union, Literal
from enum import Enum

class SupportedPredictionModel(str, Enum):
    TMBED = "TMbed"                    # Membrane topology
    VESPAG = "VespaG"                  # Variant effects
    SETH = "SETH"                      # Disorder prediction
    LIGHT_ATTENTION = "LightAttention" # Subcellular localization
    PROTT5_CONSERVATION = "ProtT5Conservation"
    PROTT5_SECSTRUCT = "ProtT5SecStruct"

class PredictionRequest(BaseModel):
    sequences: List[str] = Field(
        ...,
        description="Protein sequences for prediction",
        min_items=1,
        max_items=100  # Smaller batch for predictions
    )
    prediction_type: SupportedPredictionModel = Field(
        ...,
        description="Type of prediction to perform"
    )
    embedding_model: str = Field(
        "ProtT5",
        description="Embedding model to use as input"
    )
    include_confidence: bool = Field(
        True,
        description="Include confidence scores in results"
    )

class TMbedPredictionRequest(PredictionRequest):
    """Specialized request for membrane topology prediction"""
    prediction_type: Literal[SupportedPredictionModel.TMBED] = SupportedPredictionModel.TMBED
    topology_threshold: float = Field(0.5, ge=0.0, le=1.0)

class VespaGPredictionRequest(PredictionRequest):
    """Specialized request for variant effect prediction"""
    prediction_type: Literal[SupportedPredictionModel.VESPAG] = SupportedPredictionModel.VESPAG
    variants: List[str] = Field(..., description="Variant specifications (e.g., 'A123G')")

class BatchPredictionRequest(BaseModel):
    """For processing multiple prediction types on same sequences"""
    sequences: List[str] = Field(..., description="Protein sequences")
    prediction_types: List[SupportedPredictionModel] = Field(..., min_items=1)
    embedding_model: str = Field("ProtT5")
```

#### 2.2 Prediction Domain Service
```python
# domains/protein_predictions/services/prediction_service.py
from typing import List, Dict, Any, Optional
from ..models.prediction_request import PredictionRequest, TMbedPredictionRequest, VespaGPredictionRequest
from ..models.prediction_response import PredictionResponse, PredictionTaskResponse
from ...protein_embeddings.services.embedding_service import ProteinEmbeddingService
from ..repositories.model_repository import ModelRepository

class ProteinPredictionService:
    """Core domain service orchestrating prediction workflows"""

    def __init__(
        self,
        embedding_service: ProteinEmbeddingService,
        model_repository: ModelRepository,
        task_manager
    ):
        self.embedding_service = embedding_service
        self.model_repo = model_repository
        self.task_manager = task_manager

        # Initialize prediction-specific services
        self.tmbed_service = TMbedService(model_repository)
        self.vespag_service = VespaGService(model_repository)
        self.disorder_service = DisorderService(model_repository)
        self.localization_service = LocalizationService(model_repository)

    async def process_prediction_request(
        self,
        request: PredictionRequest,
        user_id: str
    ) -> Union[PredictionResponse, PredictionTaskResponse]:
        """Main prediction workflow: embeddings → predictions"""

        # Step 1: Ensure embeddings exist
        embeddings = await self.embedding_service.get_embeddings_for_prediction(
            sequences=request.sequences,
            model_name=request.embedding_model
        )

        # Check if we need to compute embeddings first
        if not embeddings:
            # Start embedding task, then prediction task
            task_id = await self.task_manager.enqueue_embedding_then_prediction_task(
                sequences=request.sequences,
                embedding_model=request.embedding_model,
                prediction_request=request,
                user_id=user_id
            )
            return PredictionTaskResponse(
                task_id=task_id,
                status="pending",
                message="Computing embeddings before prediction",
                workflow_stage="embedding"
            )

        # Step 2: Route to appropriate prediction service
        prediction_service = self._get_prediction_service(request.prediction_type)

        # For fast predictions, return immediately
        if prediction_service.is_fast_prediction():
            results = await prediction_service.predict(
                embeddings=embeddings,
                request=request
            )
            return PredictionResponse(
                predictions=results,
                model_used=request.prediction_type,
                embedding_model=request.embedding_model,
                cached=False
            )

        # For slow predictions, use background task
        task_id = await self.task_manager.enqueue_prediction_task(
            embeddings=embeddings,
            prediction_request=request,
            user_id=user_id
        )

        return PredictionTaskResponse(
            task_id=task_id,
            status="pending",
            message=f"Running {request.prediction_type} prediction",
            workflow_stage="prediction"
        )

    def _get_prediction_service(self, prediction_type: str):
        """Route to appropriate specialized service"""
        services_map = {
            "TMbed": self.tmbed_service,
            "VespaG": self.vespag_service,
            "SETH": self.disorder_service,
            "LightAttention": self.localization_service
        }
        return services_map.get(prediction_type)

class TMbedService:
    """Specialized service for membrane topology prediction"""

    def __init__(self, model_repository: ModelRepository):
        self.model_repo = model_repository

    def is_fast_prediction(self) -> bool:
        return True  # TMbed is relatively fast

    async def predict(
        self,
        embeddings: Dict[str, np.ndarray],
        request: TMbedPredictionRequest
    ) -> List[Dict[str, Any]]:
        """Run TMbed prediction on embeddings"""
        model = await self.model_repo.get_tmbed_model()

        results = []
        for seq_id, embedding in embeddings.items():
            # TMbed-specific prediction logic
            topology_pred = model.predict_topology(embedding)
            confidence = model.get_confidence(embedding)

            results.append({
                "sequence_id": seq_id,
                "topology": topology_pred,
                "confidence": confidence if request.include_confidence else None,
                "transmembrane_regions": model.get_tm_regions(topology_pred)
            })

        return results
```

#### 2.3 Domain Integration in FastAPI
```python
# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .middleware import UserTrackingMiddleware

# Import domain routers
from ..domains.protein_embeddings.api.embedding_router import router as embedding_router
from ..domains.protein_predictions.api.prediction_router import router as prediction_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Biocentral Server - Domain Architecture",
        description="Protein analysis server with domain-oriented architecture",
        version="0.3.0"
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(UserTrackingMiddleware)

    # CORE DOMAIN ROUTERS (Priority 1)
    app.include_router(
        embedding_router,
        prefix="/api/v1/embeddings",
        tags=["Protein Embeddings"]
    )
    app.include_router(
        prediction_router,
        prefix="/api/v1/predictions",
        tags=["Protein Predictions"]
    )

    # Core workflow endpoint - embeddings + predictions combined
    @app.post("/api/v1/analyze", tags=["Core Workflow"])
    async def analyze_proteins(
        sequences: List[str],
        prediction_types: List[str],
        embedding_model: str = "ProtT5"
    ):
        """One-shot endpoint for complete protein analysis workflow"""
        # This would orchestrate embeddings → predictions
        pass

    return app

app = create_app()
```

## Major Roadblocks and Domain-Specific Challenges

### 1. **Embedding-Prediction Workflow Coordination** (Critical)
- **Problem**: Current Flask system handles embedding → prediction pipeline through separate endpoints
- **Domain Impact**: Core workflow requires coordination between embedding and prediction domains
- **Solution**: Implement domain service orchestration with workflow state management
- **Complexity**: High - requires careful async task coordination

### 2. **Biotrainer Decoupling in Domain Context** (High Priority)
- **Problem**: Biotrainer logic is embedded throughout prediction models
- **Current Files**: `prediction_models/biotrainer_task.py`, `predict/models/*/`
- **Domain Solution**: Create ML framework adapter within prediction domain
- **Complexity**: High - affects core prediction functionality

### 3. **Background Task Domain Integration** (Critical)
- **Problem**: Redis Queue (RQ) tasks must work across domain boundaries
- **Domain Impact**: Embedding tasks must trigger prediction tasks seamlessly
- **Solution**: Domain-aware task manager with workflow orchestration
- **Complexity**: Medium - RQ system needs domain-specific routing

### 4. **Cross-Domain Data Flow** (Medium Priority)
- **Problem**: Embeddings computed in one domain must be accessible to prediction domain
- **Current Pattern**: Direct database access across modules
- **Domain Solution**: Well-defined domain interfaces and data contracts
- **Complexity**: Medium - requires careful API design

### 5. **Model Loading and Caching** (Medium Priority)
- **Problem**: TMbed, VespaG, and other models need efficient loading/caching
- **Current Files**: `predict/models/base_model/`, `predict/predict_initializer.py`
- **Domain Solution**: Domain-specific model repositories with caching strategies
- **Complexity**: Medium - performance-critical for prediction latency

## Core Workflow Test Cases for Domain Migration

### 1. **Embedding Domain Functionality Tests**

#### 1.1 Single Sequence Embedding
```python
def test_single_sequence_embedding(client, auth_headers):
    """Test basic embedding functionality"""
    request_data = {
        "sequences": ["MKLLVLGLSG"],
        "model_name": "ProtT5",
        "return_raw_embeddings": True
    }

    response = client.post(
        "/api/v1/embeddings/embed",
        json=request_data,
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    # Check for immediate response or task creation
    if "embeddings" in data:
        assert len(data["embeddings"]) == 1
        assert data["model_used"] == "ProtT5"
    else:
        assert "task_id" in data
        assert data["status"] == "pending"
```

#### 1.2 Batch Embedding Processing
```python
def test_batch_embedding_processing(client):
    """Test batch embedding with cache checking"""
    sequences = ["MKLLVLGLSG", "MKLLVLGLSGAA", "UNKNOWN_SEQUENCE"]

    # First, check missing embeddings
    missing_response = client.post(
        "/api/v1/embeddings/missing",
        json={"sequences": sequences, "model_name": "ProtT5"}
    )

    assert missing_response.status_code == 200
    missing_data = missing_response.json()
    assert "missing_sequences" in missing_data

    # Then request embeddings for missing ones
    if missing_data["missing_sequences"]:
        embed_response = client.post(
            "/api/v1/embeddings/embed",
            json={
                "sequences": missing_data["missing_sequences"],
                "model_name": "ProtT5"
            }
        )
        assert embed_response.status_code == 200
```

### 2. **Prediction Domain Functionality Tests**

#### 2.1 TMbed Membrane Topology Prediction
```python
def test_tmbed_prediction_workflow(client, auth_headers):
    """Test complete TMbed prediction workflow"""
    request_data = {
        "sequences": ["MKLLVLGLSGAGAGAGAGAG"],  # Membrane-like sequence
        "prediction_type": "TMbed",
        "embedding_model": "ProtT5",
        "topology_threshold": 0.6
    }

    response = client.post(
        "/api/v1/predictions/predict",
        json=request_data,
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    # Check for immediate response or task creation
    if "predictions" in data:
        # Fast prediction completed
        assert len(data["predictions"]) == 1
        prediction = data["predictions"][0]
        assert "topology" in prediction
        assert "transmembrane_regions" in prediction
        assert data["model_used"] == "TMbed"
    else:
        # Background task created
        assert "task_id" in data
        assert data["workflow_stage"] in ["embedding", "prediction"]
```

#### 2.2 VespaG Variant Effect Prediction
```python
def test_vespag_prediction_workflow(client):
    """Test VespaG variant effect prediction"""
    request_data = {
        "sequences": ["MKLLVLGLSG"],
        "prediction_type": "VespaG",
        "variants": ["M1A", "K2R", "L3P"],
        "embedding_model": "ProtT5"
    }

    response = client.post(
        "/api/v1/predictions/predict",
        json=request_data
    )

    assert response.status_code == 200
    data = response.json()

    if "predictions" in data:
        prediction = data["predictions"][0]
        assert "variant_effects" in prediction
        assert len(prediction["variant_effects"]) == 3
    else:
        assert "task_id" in data
```

### 3. **Cross-Domain Workflow Integration Tests**

#### 3.1 Embedding → Prediction Pipeline
```python
def test_embedding_prediction_pipeline(client):
    """Test the complete embeddings → predictions workflow"""

    # Step 1: Start with fresh sequences (no cached embeddings)
    sequences = [f"MKLLVLGLSG{i}" for i in range(3)]  # Unique sequences

    # Step 2: Request prediction (should trigger embedding first)
    prediction_request = {
        "sequences": sequences,
        "prediction_type": "TMbed",
        "embedding_model": "ProtT5"
    }

    prediction_response = client.post(
        "/api/v1/predictions/predict",
        json=prediction_request
    )

    assert prediction_response.status_code == 200
    pred_data = prediction_response.json()

    if "task_id" in pred_data:
        task_id = pred_data["task_id"]

        # Step 3: Check task status progression
        # Should go: embedding → prediction → completed
        status_response = client.get(f"/api/v1/tasks/{task_id}/status")
        assert status_response.status_code == 200

        # Step 4: Verify embeddings are now cached
        embedding_check = client.post(
            "/api/v1/embeddings/missing",
            json={"sequences": sequences, "model_name": "ProtT5"}
        )
        # Should have fewer or no missing embeddings after task completion
```

#### 3.2 Batch Multi-Prediction Workflow
```python
def test_batch_multi_prediction(client):
    """Test running multiple prediction types on same sequences"""
    request_data = {
        "sequences": ["MKLLVLGLSG", "AAAAGGGGCCCC"],
        "prediction_types": ["TMbed", "SETH", "LightAttention"],
        "embedding_model": "ProtT5"
    }

    response = client.post(
        "/api/v1/predictions/batch_predict",
        json=request_data
    )

    assert response.status_code == 200
    data = response.json()

    if "predictions" in data:
        # All predictions completed immediately
        assert len(data["predictions"]) == 2  # 2 sequences
        for pred in data["predictions"]:
            assert len(pred["results"]) == 3  # 3 prediction types
    else:
        # Background processing
        assert "task_id" in data
```

### 4. **Domain Performance and Load Tests**

#### 4.1 Concurrent Embedding Requests
```python
def test_concurrent_embedding_requests(client):
    """Test domain services handle concurrent requests"""
    import asyncio
    import aiohttp

    async def make_embedding_request(session, sequence_id):
        request_data = {
            "sequences": [f"MKLLVLGLSG{sequence_id}"],
            "model_name": "ProtT5"
        }
        async with session.post(
            "http://localhost:8000/api/v1/embeddings/embed",
            json=request_data
        ) as response:
            return await response.json()

    async def test_concurrent():
        async with aiohttp.ClientSession() as session:
            tasks = [
                make_embedding_request(session, i)
                for i in range(50)
            ]
            results = await asyncio.gather(*tasks)

            # All requests should succeed
            assert all("task_id" in result or "embeddings" in result for result in results)

    asyncio.run(test_concurrent())
```

#### 4.2 Domain Service Resource Management
```python
def test_domain_service_resource_management(client):
    """Test that domain services properly manage ML model resources"""

    # Make multiple prediction requests
    for prediction_type in ["TMbed", "VespaG", "SETH"]:
        response = client.post(
            "/api/v1/predictions/predict",
            json={
                "sequences": ["MKLLVLGLSG"],
                "prediction_type": prediction_type
            }
        )
        assert response.status_code == 200

    # Check memory usage doesn't grow indefinitely
    # (This would be monitored via system metrics)
```

### 5. **Domain Migration Validation Tests**

#### 5.1 Feature Parity Validation
```python
def test_feature_parity_with_flask_version():
    """Ensure all Flask functionality is preserved in domain architecture"""

    # Test all supported embedding models
    embedding_models = ["ProtT5", "ESM-2", "Ankh"]
    for model in embedding_models:
        response = client.post(
            "/api/v1/embeddings/embed",
            json={"sequences": ["MKLLVLGLSG"], "model_name": model}
        )
        assert response.status_code == 200

    # Test all supported prediction models
    prediction_models = ["TMbed", "VespaG", "SETH", "LightAttention"]
    for model in prediction_models:
        response = client.post(
            "/api/v1/predictions/predict",
            json={"sequences": ["MKLLVLGLSG"], "prediction_type": model}
        )
        assert response.status_code == 200
```

#### 5.2 Background Task System Validation
```python
def test_background_task_system_still_works():
    """Ensure Redis Queue system works with domain architecture"""

    # Submit a complex task that requires background processing
    response = client.post(
        "/api/v1/predictions/predict",
        json={
            "sequences": ["A" * 1000],  # Long sequence requiring background processing
            "prediction_type": "VespaG"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data

    # Verify task appears in Redis queue
    # (This would check Redis directly or via monitoring endpoint)
```

## Domain Migration Timeline and Phases

### **Phase 1: Core Embeddings Domain (Weeks 1-3)**
**Priority**: Highest - Foundation for all predictions

#### Week 1: Domain Structure Setup
- [ ] Create domain-oriented project structure
- [ ] Set up FastAPI application with embeddings domain
- [ ] Implement embedding Pydantic models with validation
- [ ] Create embedding repository layer

#### Week 2: Embedding Service Implementation
- [ ] Implement `ProteinEmbeddingService` with workflow orchestration
- [ ] Migrate embedding computation logic from Flask blueprints
- [ ] Integrate with existing Redis Queue (RQ) system
- [ ] Add embedding caching and retrieval logic

#### Week 3: Testing and Integration
- [ ] Write comprehensive embedding domain tests
- [ ] Test embedding workflow: cache check → computation → storage
- [ ] Performance testing for concurrent embedding requests
- [ ] Integration with existing database layer

### **Phase 2: Core Predictions Domain (Weeks 4-6)**
**Priority**: Highest - Core business value

#### Week 4: Prediction Models and Services
- [ ] Create prediction domain structure with specialized services
- [ ] Implement TMbed service for membrane topology prediction
- [ ] Implement VespaG service for variant effect prediction
- [ ] Create prediction Pydantic models with type-specific validation

#### Week 5: Cross-Domain Integration
- [ ] Implement embedding → prediction workflow orchestration
- [ ] Create domain service communication patterns
- [ ] Migrate existing prediction models (SETH, LightAttention, etc.)
- [ ] Test background task coordination between domains

#### Week 6: Prediction Workflow Testing
- [ ] Test complete embedding → prediction pipelines
- [ ] Validate TMbed, VespaG, and other model functionality
- [ ] Performance testing for prediction workflows
- [ ] Cross-domain data flow validation

### **Phase 3: Supporting Domains (Weeks 7-8)**
**Priority**: Medium - Enhanced functionality

#### Week 7: Model Training Domain (Optional)
- [ ] Migrate biotrainer integration to domain architecture
- [ ] Create ML framework adapter pattern
- [ ] Implement training workflow orchestration
- [ ] Test model training and evaluation pipelines

#### Week 8: Analysis and Task Management Domains
- [ ] Create protein analysis domain (sequence similarity, etc.)
- [ ] Enhance task management for cross-domain workflows
- [ ] Implement domain-aware monitoring and logging
- [ ] Final integration testing and optimization

### **Phase 4: Production Deployment (Weeks 9-10)**
**Priority**: Critical - Go-live preparation

#### Week 9: Production Readiness
- [ ] Performance benchmarking vs Flask version
- [ ] Security review and authentication integration
- [ ] Docker container updates for domain architecture
- [ ] Monitoring and observability setup

#### Week 10: Deployment and Validation
- [ ] Staged deployment with traffic switching
- [ ] Production validation testing
- [ ] Performance monitoring and optimization
- [ ] Documentation and knowledge transfer

## Success Criteria for Domain Migration

### **Core Workflow Success Criteria**

#### **Functional Requirements**
- [ ] **Embedding Generation**: All protein sequences can be embedded using ProtT5, ESM-2, Ankh models
- [ ] **Prediction Accuracy**: TMbed, VespaG, SETH, LightAttention predictions match Flask version output
- [ ] **Workflow Integration**: Embedding → prediction pipeline works seamlessly with automatic caching
- [ ] **Background Processing**: Complex tasks use Redis Queue with proper progress tracking
- [ ] **Data Persistence**: Embeddings are properly stored and retrieved from PostgreSQL database

#### **Performance Requirements**
- [ ] **Response Time**: API response times ≤ Flask version (embedding requests <200ms, predictions <500ms)
- [ ] **Throughput**: Handle ≥100 concurrent embedding requests without degradation
- [ ] **Memory Usage**: Memory consumption ≤ Flask version for equivalent workloads
- [ ] **Task Processing**: Background tasks complete within 110% of original Flask timing

#### **Quality Requirements**
- [ ] **Type Safety**: All API requests/responses use Pydantic models with comprehensive validation
- [ ] **Error Handling**: Consistent error responses with appropriate HTTP status codes
- [ ] **Documentation**: Auto-generated OpenAPI documentation covers all endpoints
- [ ] **Test Coverage**: ≥90% test coverage for domain services and API endpoints

### **Domain Architecture Success Criteria**

#### **Domain Isolation**
- [ ] **Embedding Domain**: Fully self-contained with clear interfaces to other domains
- [ ] **Prediction Domain**: Independent of embedding implementation details
- [ ] **Cross-Domain Communication**: Well-defined contracts between domains
- [ ] **Business Logic Separation**: Domain services contain no web framework dependencies

#### **Technical Debt Reduction**
- [ ] **Biotrainer Decoupling**: ML framework logic abstracted into adapter pattern
- [ ] **Import Simplification**: Eliminated relative imports and circular dependencies
- [ ] **Configuration Management**: Centralized configuration with dependency injection
- [ ] **Code Maintainability**: Reduced cyclomatic complexity and improved modularity

### **Migration Validation Criteria**

#### **Feature Parity Validation**
```python
# Comprehensive validation test
def test_complete_workflow_parity():
    """Validate that domain architecture produces identical results to Flask version"""

    test_sequences = ["MKLLVLGLSG", "AAAAGGGGCCCC", "MEMBRANE_PROTEIN_SEQUENCE"]

    # Test each embedding model
    for embedding_model in ["ProtT5", "ESM-2", "Ankh"]:
        # Test each prediction model
        for prediction_model in ["TMbed", "VespaG", "SETH", "LightAttention"]:
            response = client.post("/api/v1/analyze", json={
                "sequences": test_sequences,
                "prediction_types": [prediction_model],
                "embedding_model": embedding_model
            })

            assert response.status_code == 200
            # Validate against known good results from Flask version
```

#### **Performance Regression Testing**
```python
def test_performance_regression():
    """Ensure performance meets or exceeds Flask baseline"""

    import time
    start_time = time.time()

    # Run standardized performance test
    response = client.post("/api/v1/embeddings/embed", json={
        "sequences": ["PROTEIN_SEQUENCE"] * 100,
        "model_name": "ProtT5"
    })

    end_time = time.time()
    response_time = end_time - start_time

    # Should be within 10% of Flask baseline
    assert response_time <= FLASK_BASELINE_TIME * 1.1
```

### **Deployment Readiness Criteria**

#### **Production Deployment Checklist**
- [ ] **Container Updates**: Docker images updated with FastAPI and domain architecture
- [ ] **Environment Configuration**: All environment variables mapped to new domain structure
- [ ] **Database Migrations**: Any schema changes applied and tested
- [ ] **Monitoring Setup**: Application metrics, logging, and health checks configured
- [ ] **Security Review**: Authentication, authorization, and input validation verified
- [ ] **Load Testing**: Production-level load testing completed successfully
- [ ] **Rollback Plan**: Clear rollback procedure documented and tested
- [ ] **Documentation**: Updated API documentation and deployment guides

#### **Go-Live Success Metrics**
- [ ] **Zero Downtime**: Migration completes without service interruption
- [ ] **Error Rate**: <0.1% error rate in first 24 hours post-deployment
- [ ] **Performance**: Response times within 5% of pre-migration baseline
- [ ] **User Experience**: No reported functionality regressions from users
- [ ] **System Stability**: No memory leaks or resource exhaustion after 72 hours

This restructured migration plan prioritizes the core protein analysis workflow (embeddings → predictions) with a domain-oriented architecture approach, focusing on TMbed, VespaG, and other embedding-based predictors as the primary business value drivers.
