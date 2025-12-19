```mermaid
flowchart TD
    classDef data fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000;
    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000;
    classDef model fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000;
    classDef audit fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#000;
    classDef output fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000;

    Raw[Raw Transactions CSV]:::data --> Pivot[Data Pivoting: User x Time]:::process

    subgraph FeatureEng ["Feature Engineering Layer"]
        direction TB
        Pivot --> SplitLogic{Feature Logic}
        
        SplitLogic -->|Aggregation| Baseline["Baseline Features<br/>(Mean/Max/Sum)"]:::process
        
        SplitLogic -->|JIT Compilation| Numba[Numba Feature Engine]:::process
        
        Numba --> Feat1[Consecutive Late Streaks]
        Numba --> Feat2["Payment Velocity / Slope"]
        Numba --> Feat3[Critical Utilization]
        
        UnitTest["pytest: Unit Testing &<br/>Edge Case Validation"]:::audit -.->|Validates| Numba
    end

    Feat1 & Feat2 & Feat3 & Baseline --> Merged[Engineered Feature Table]:::data

    subgraph Modeling ["Modeling & Experimentation"]
        Merged --> SplitData[Train / Test Split]
        SplitData --> Train[Model Training]:::model
        
        Train --> LR["Logistic Regression<br/>(Interpretability)"]:::model
        Train --> LGBM["LightGBM<br/>(Non-Linear)"]:::model
    end

    subgraph Eval ["Evaluation & Audit"]
        LR & LGBM --> Metrics["Performance Metrics<br/>(AUC / ROC)"]:::output
        
        SplitData --> PSI["PSI Check<br/>(Population Stability Index)"]:::audit
        LGBM -.->|Probabilities| PSI
        PSI --> PSIStatus{Is Stable?}
    end

    subgraph Impact ["Impact Study"]
        Metrics --> Scenarios
        Scenarios --> S1[Scenario A: Baseline Only]
        Scenarios --> S2[Scenario B: Behavior Only]
        Scenarios --> S3[Scenario C: Hybrid Model]:::output
    end

    S1 & S2 & S3 --> Uplift[Final Verdict & Uplift Analysis]:::output

    linkStyle default stroke:#333,stroke-width:1px;