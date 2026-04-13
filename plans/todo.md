--- DOING
 - Generating the embedding | Colab
 - Run ablation study | M4 pro
 - Generate other baselines | M2 Pro 
 - Search bracis & Create theses & report & abstract
   - **Block: Florida analyzes**
   - **Block: study over the baselines** 
 - 

--- MUST DO
1. Get Baselines
2. Abalation study - Fusio+Alabama && Start to write and search for the bracis [Paralell]
3. Full execution && justify data leckage
4. Execute flops and wall test to compare mtlnet
--- FOLLOW UP
5. Analise de variancia de analise the hipotese estatistica
6. Justify data leckage
--- OPC
4. Do we need to save the embedding model to validate in the mtlnet train ?
5. Implemented new datasets
6. Get baselines witht the new data




python scripts/train.py --task mtl --state florida --engine hgi \
    --epochs 50 --folds 5 --model mtlnet_cgc \
    --mtl-loss equal_weight --seed 42 --embedding-dim 64 \
    --model-param num_shared_experts=2 \
    --model-param num_task_experts=2
                                           