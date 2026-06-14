# References

> Consolidated from the 2026-06-12 literature survey. Grouped by topic; all links verified at survey time. Items marked ⚠ have a caveat noted inline.

## Next-POI recommendation

1. Liu, Wu, Wang, Tan. "Predicting the Next Location: A Recurrent Model with Spatial and Temporal Contexts" (ST-RNN). AAAI 2016. https://ojs.aaai.org/index.php/AAAI/article/view/9971
2. Feng et al. "DeepMove: Predicting Human Mobility with Attentional Recurrent Networks". WWW 2018. https://dl.acm.org/doi/10.1145/3178876.3186058
3. Yang, Fankhauser, Rosso, Cudré-Mauroux. "Location Prediction over Sparse User Mobility Traces Using RNNs: Flashback in Hidden States!". IJCAI 2020. https://www.ijcai.org/proceedings/2020/302
4. Sun et al. "Where to Go Next: Modeling Long- and Short-Term User Preferences for Point-of-Interest Recommendation" (LSTPM). AAAI 2020. https://ojs.aaai.org/index.php/AAAI/article/view/5353
5. ⚠ GeoSAN — Lian et al., KDD 2020. Exact title to be verified on DBLP before citing (located via secondary sources only).
6. Luo, Liu, Liu. "STAN: Spatio-Temporal Attention Network for Next Location Recommendation". WWW 2021. https://dl.acm.org/doi/10.1145/3442381.3449998 · https://arxiv.org/abs/2102.04095
7. Yang, Liu, Zhao. "GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation". SIGIR 2022. https://dl.acm.org/doi/10.1145/3477495.3531983 · https://github.com/songyangco/GETNext
8. Rao et al. "Graph-Flashback Network for Next Location Recommendation". KDD 2022. https://dl.acm.org/doi/10.1145/3534678.3539383
9. Wang, Zhu, Liu, Wang. "Learning Graph-based Disentangled Representations for Next POI Recommendation" (DRAN). SIGIR 2022. https://dl.acm.org/doi/10.1145/3477495.3532012
10. Yin et al. "Next POI Recommendation with Dynamic Graph and Explicit Dependency" (SNPM). AAAI 2023. https://ojs.aaai.org/index.php/AAAI/article/view/25608
11. Wang et al. "Adaptive Graph Representation Learning for Next POI Recommendation" (AGRAN). SIGIR 2023. https://dl.acm.org/doi/10.1145/3539618.3591634
12. Yan et al. "Spatio-Temporal Hypergraph Learning for Next POI Recommendation" (STHGCN). SIGIR 2023. https://dl.acm.org/doi/10.1145/3539618.3591770 · https://github.com/alipay/Spatio-Temporal-Hypergraph-Model
13. Lai et al. "Disentangled Contrastive Hypergraph Learning for Next POI Recommendation" (DCHL). SIGIR 2024. https://dl.acm.org/doi/10.1145/3626772.3657726
14. Feng et al. "ROTAN: A Rotation-based Temporal Attention Network for Time-Specific Next POI Recommendation". KDD 2024. https://dl.acm.org/doi/10.1145/3637528.3671809
15. Duan et al. "CLSPRec". CIKM 2023. https://dl.acm.org/doi/10.1145/3583780.3614813
16. "CLLP: Contrastive Learning Framework Based on Latent Preferences for Next POI Recommendation". SIGIR 2024. https://dl.acm.org/doi/10.1145/3626772.3657730
17. Qin et al. "A Diffusion Model for POI Recommendation" (Diff-POI). ACM TOIS 42(2), 2023. https://dl.acm.org/doi/10.1145/3624475
18. Li et al. "Large Language Models for Next Point-of-Interest Recommendation" (LLM4POI). SIGIR 2024. https://dl.acm.org/doi/10.1145/3626772.3657840
19. Feng et al. "Where to Move Next: Zero-shot Generalization of LLMs for Next POI Recommendation" (LLMMove). IEEE CAI 2024. https://arxiv.org/abs/2404.01855
20. Liu et al. "NextLocLLM". 2024. https://arxiv.org/abs/2410.09129
21. Feng et al. "AgentMove". NAACL 2025. https://aclanthology.org/2025.naacl-long.61/
22. Wongso et al. "GenUP". SIGSPATIAL 2025. https://arxiv.org/abs/2410.20643
23. "Refine-POI: Reinforcement Fine-Tuned LLMs for Next POI Recommendation". 2025. https://arxiv.org/abs/2506.21599
24. "Generative Next POI Recommendation with Semantic ID" (GNPR-SID). KDD 2025. https://dl.acm.org/doi/10.1145/3711896.3736981
25. "TOOL4POI". 2025. https://arxiv.org/abs/2511.06405
26. "Multifaceted Scenario-Aware Hypergraph Learning" (MSHL). AAAI 2026. https://arxiv.org/abs/2601.11610
27. "ASTHN". ISPRS IJGI 15(6):242, 2026. https://doi.org/10.3390/ijgi15060242
28. Li et al. "ReHDM" (region-aware dual-level hypergraph). IJCAI 2025. (Reimplemented in-repo; see `docs/baselines/next_region/comparison.md`.)
29. Massive-STEPS benchmark. 2025. https://arxiv.org/abs/2505.11239 · https://github.com/cruiseresearchgroup/Massive-STEPS

## Next-category / activity prediction & MTL for mobility

30. Ye, Zhu, Cheng. "What's Your Next Move: User Activity Prediction in LBSNs". SDM 2013. https://www1.se.cuhk.edu.hk/~hcheng/paper/sdm2013.pdf
31. He, Li, Liao. "Category-aware Next POI Recommendation via Listwise BPR". IJCAI 2017. https://www.ijcai.org/proceedings/2017/255
32. Liao et al. "Predicting Activity and Location with Multi-task Context Aware Recurrent Neural Network" (MCARNN). IJCAI 2018. https://www.ijcai.org/proceedings/2018/477
33. Zhang et al. "Modeling hierarchical category transition for next POI recommendation with uncertain check-ins" (HCT). Information Sciences 2019.
34. Yu et al. "A Category-Aware Deep Model for Successive POI Recommendation on Sparse Check-in Data" (CatDM). WWW 2020. https://dl.acm.org/doi/10.1145/3366423.3380202
35. Zhang et al. "An Interactive Multi-Task Learning Framework for Next POI Recommendation with Uncertain Check-ins" (iMTL). IJCAI 2020. https://www.ijcai.org/proceedings/2020/491 · https://github.com/iMTL2020/iMTL
36. Lim et al. "Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation" (HMT-GRN). SIGIR 2022. https://dl.acm.org/doi/10.1145/3477495.3531989 · https://github.com/poi-rec/HMT-GRN ; journal extension: ACM TORS 2023, https://dl.acm.org/doi/10.1145/3610584
37. Huang, Xu et al. "Human Mobility Prediction with Causal and Spatial-constrained Multi-task Network" (CSLSL). EPJ Data Science 13, 2024. https://link.springer.com/article/10.1140/epjds/s13688-024-00460-7 · https://arxiv.org/abs/2206.05731
38. Xue, Salim et al. "MobTCast". NeurIPS 2021. https://arxiv.org/abs/2110.01401
39. He et al. "ImNext". Knowledge-Based Systems 2024. https://www.sciencedirect.com/science/article/abs/pii/S0950705124003095
40. "MSAN". ISPRS IJGI 12(7):297, 2023. https://www.mdpi.com/2220-9964/12/7/297
41. Sun et al. "Going Where, by Whom, and at What Time" (MCLP). KDD 2024. https://dl.acm.org/doi/10.1145/3637528.3671916
42. Wang et al. "Hierarchy aware-based multi-task learning for user location prediction" (HAMTL). J. Supercomputing 81:1196, 2025. https://link.springer.com/article/10.1007/s11227-025-07643-7
43. "MMPAN". Expert Systems with Applications 2024. https://www.sciencedirect.com/science/article/abs/pii/S0957417424030562
44. "KGTB: Knowledge Graph Tokenization for Behavior-Aware Generative Next POI Recommendation". 2025. https://arxiv.org/abs/2509.12350
45. Lai et al. "SPENT+". IEEE APNOMS 2021. https://ieeexplore.ieee.org/document/9562645
46. "Predicting a Person's Next Activity Region with DRRGNN". ACM TKDD 16(6), 2022. https://dl.acm.org/doi/10.1145/3529091
47. "Where and When: Predict Next POI and Its Explicit Timestamp". IJCAI 2025. https://www.ijcai.org/proceedings/2025/0390.pdf
48. "Joint prediction of travel mode choice and purpose". Travel Behaviour and Society 2024. https://www.sciencedirect.com/science/article/pii/S2214367X23000765
49. "United Prediction of Travel Modes and Purposes in Travel Chains". Mathematics 13:1528, 2025. https://doi.org/10.3390/math13091528
50. "CHA: Categorical Hierarchy-based Attention". ACM TOIT 2021. https://dl.acm.org/doi/fullHtml/10.1145/3464300

## MTL architectures & optimization

51. Caruana. "Multitask Learning". Machine Learning 28, 1997. https://link.springer.com/article/10.1023/A:1007379606734
52. Misra et al. "Cross-stitch Networks". CVPR 2016. https://arxiv.org/abs/1604.03539
53. Ma et al. "MMoE". KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3220007
54. Tang et al. "Progressive Layered Extraction" (PLE/CGC). RecSys 2020. https://dl.acm.org/doi/10.1145/3383313.3412236
55. Hazimeh et al. "DSelect-k". NeurIPS 2021. https://arxiv.org/abs/2106.03760
56. Liu, Johns, Davison. "End-to-End Multi-Task Learning with Attention" (MTAN; DWA). CVPR 2019. https://arxiv.org/abs/1803.10704
57. Perez et al. "FiLM". AAAI 2018. https://arxiv.org/abs/1709.07871
58. Tsai et al. "Multimodal Transformer for Unaligned Multimodal Language Sequences" (MulT). ACL 2019. https://arxiv.org/abs/1906.00295
59. Kendall, Gal, Cipolla. Uncertainty weighting. CVPR 2018. https://arxiv.org/abs/1705.07115
60. Chen et al. "GradNorm". ICML 2018. https://arxiv.org/abs/1711.02257
61. Yu et al. "Gradient Surgery" (PCGrad). NeurIPS 2020. https://arxiv.org/abs/2001.06782
62. Liu et al. "CAGrad". NeurIPS 2021. https://arxiv.org/abs/2110.14048
63. Navon et al. "Multi-Task Learning as a Bargaining Game" (Nash-MTL). ICML 2022. https://arxiv.org/abs/2202.01017
64. Senushkin et al. "Aligned-MTL". CVPR 2023. https://arxiv.org/abs/2305.19000
65. Standley et al. "Which Tasks Should Be Learned Together?". ICML 2020. https://arxiv.org/abs/1905.07553
66. Xin et al. "Do Current Multi-Task Optimization Methods Even Help?". NeurIPS 2022. https://arxiv.org/abs/2209.11379
67. Kurin et al. "In Defense of the Unitary Scalarization". NeurIPS 2022. https://arxiv.org/abs/2201.04122
68. Zhang et al. "A Survey on Negative Transfer". IEEE/CAA JAS 2023. https://arxiv.org/abs/2009.00909
69. Lin et al. "Focal Loss". ICCV 2017. https://arxiv.org/abs/1708.02002

## Location / check-in / trajectory embeddings

70. Veličković et al. "Deep Graph Infomax" (DGI). ICLR 2019. https://arxiv.org/abs/1809.10341
71. Huang, Zhang, Mai, Guo, Cui. "Learning urban region representations with POIs and hierarchical graph infomax" (HGI). ISPRS JPRS 196:134–145, 2023. https://www.sciencedirect.com/science/article/abs/pii/S0924271622003148 · https://github.com/RightBank/HGI
72. Feng et al. "POI2Vec". AAAI 2017. https://ojs.aaai.org/index.php/AAAI/article/view/10500
73. Zhao et al. "Geo-Teaser". WWW 2017 Companion. https://link.springer.com/chapter/10.1007/978-981-13-1349-3_4
74. Chang et al. "CAPE". IJCAI 2018. https://www.ijcai.org/proceedings/2018/458
75. Wan, Lin, Guo, Lin. "TALE: Pre-Training Time-Aware Location Embeddings". IEEE TKDE 34(11), 2022. https://ieeexplore.ieee.org/document/9351627/ · https://github.com/Logan-Lin/TALE
76. Lin, Wan, Guo, Lin. "Pre-training Context and Time Aware Location Embeddings from Spatial-Temporal Trajectories for User Next Location Prediction" (CTLE). AAAI 2021. https://ojs.aaai.org/index.php/AAAI/article/view/16548 · https://github.com/Logan-Lin/CTLE
77. Shimizu et al. "Hier: fine-grained place embeddings with spatial hierarchy". SIGSPATIAL 2020. https://arxiv.org/pdf/2002.02058
78. "CASTLE". ISPRS Archives XLVIII-4/W2, 2023. https://ui.adsabs.harvard.edu/abs/2023ISPAr48W2...15C/abstract
79. Park et al. "Geo-Tokenizer". ECML PKDD 2023. https://arxiv.org/abs/2310.01252
80. Gong et al. "CACSR: Contrastive Pre-training with Adversarial Perturbations for Check-In Sequence Representation". AAAI 2023. https://ojs.aaai.org/index.php/AAAI/article/view/25546
81. Gong et al. "STCCR". IEEE TKDE 2024. https://arxiv.org/abs/2407.15899
82. "UniTE: A Survey and Unified Pipeline for Pre-Training Spatiotemporal Trajectory Embeddings". IEEE TKDE 37(3), 2025. https://arxiv.org/abs/2407.12550 · https://github.com/Logan-Lin/UniTE
83. Yang et al. "LBSN2Vec". WWW 2019. https://dl.acm.org/doi/fullHtml/10.1145/3308558.3313635 ; "LBSN2Vec++", IEEE TKDE 2020, https://ieeexplore.ieee.org/abstract/document/9099985
84. Mai et al. "Space2Vec". ICLR 2020. https://arxiv.org/abs/2003.00824
85. Mai et al. "Sphere2Vec". ISPRS JPRS 202, 2023. https://arxiv.org/abs/2306.17624
86. Kazemi et al. "Time2Vec". 2019. https://arxiv.org/abs/1907.05321
87. ⚠ Chen et al. "Modeling Spatial Trajectories With Attribute Representation Learning" (HMRM = **Human** Mobility Representation Model — repo docs currently expand it incorrectly as "Heterogeneous"). IEEE TKDE, DOI 10.1109/TKDE.2020.3001025. https://ieeexplore.ieee.org/document/9112685/
88. Hou et al. "GraphMAE". KDD 2022 (T4.1 lever provenance). https://arxiv.org/abs/2205.10803
89. Grover, Leskovec. "node2vec". KDD 2016. https://arxiv.org/abs/1607.00653
90. Oord et al. "Representation Learning with Contrastive Predictive Coding" (InfoNCE). 2018. https://arxiv.org/abs/1807.03748

## Region representation learning

91. "ReMVC: Region Embedding With Intra and Inter-View Contrastive Learning". IEEE TKDE 2022. https://ieeexplore.ieee.org/document/9973276/
92. Wu et al. "MGFN". IJCAI 2022. https://www.ijcai.org/proceedings/2022/321
93. "ReCP". AAAI 2024. https://arxiv.org/abs/2312.09681
94. "VecCity" (benchmark incl. HGI). PVLDB 2025. https://arxiv.org/pdf/2411.00874

## MTL frontier 2023–2026 (added 2026-06-12; see `mtl_frontier.md`)

101. Liu et al. "FAMO: Fast Adaptive Multitask Optimization". NeurIPS 2023. https://arxiv.org/abs/2306.03792
102. Royer et al. "Scalarization for Multi-Task and Multi-Domain Learning at Scale". NeurIPS 2023. https://arxiv.org/pdf/2310.08910
103. Hu et al. "Revisiting Scalarization in MTL: A Theoretical Perspective". NeurIPS 2023. https://arxiv.org/abs/2308.13985
104. Achituve et al. "Bayesian Uncertainty for Gradient Aggregation in MTL" (BayesAgg-MTL). ICML 2024. https://proceedings.mlr.press/v235/achituve24a.html
105. Lin et al. "Smooth Tchebycheff Scalarization for Multi-Objective Optimization". ICML 2024. https://arxiv.org/pdf/2402.19078
106. Ban, Ji. "Fair Resource Allocation in MTL" (FairGrad). ICML 2024. https://arxiv.org/pdf/2402.15638
107. Mueller, Dredze, Andrews. "Can Optimization Trajectories Explain Multi-Task Transfer?". TMLR 2025. https://arxiv.org/pdf/2408.14677
108. "Multitask Learning 1997–2024" retrospective. HDSR 2025. https://hdsr.mitpress.mit.edu/pub/7fcc3jhv
109. Fifty et al. "Efficiently Identifying Task Groupings for MTL" (TAG). NeurIPS 2021. https://arxiv.org/abs/2109.04617
110. Navon et al. "Auxiliary Learning by Implicit Differentiation" (AuxiLearn). ICLR 2021. https://arxiv.org/abs/2007.02693
111. Liu et al. "Auto-Lambda: Disentangling Dynamic Task Relationships". TMLR 2022. https://arxiv.org/abs/2202.03091
112. Jiang et al. "ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning". NeurIPS 2023. https://arxiv.org/pdf/2301.12618 · https://github.com/thuml/ForkMerge
113. Ilharco et al. "Editing Models with Task Arithmetic". ICLR 2023. https://arxiv.org/abs/2212.04089
114. Yadav et al. "TIES-Merging". NeurIPS 2023. https://arxiv.org/abs/2306.01708
115. Yu et al. "DARE". ICML 2024. https://arxiv.org/abs/2311.03099
116. Yang et al. "AdaMerging". ICLR 2024. https://arxiv.org/abs/2310.02575
117. Ortiz-Jimenez et al. "Task Arithmetic in the Tangent Space" (weight disentanglement). NeurIPS 2023 oral. https://arxiv.org/abs/2305.12827
118. Stoica et al. "ZipIt! Merging Models from Different Tasks without Training". ICLR 2024. https://arxiv.org/pdf/2305.03053
119. "SIMO: Single-Input Multi-Output Model Merging". 2025. https://arxiv.org/abs/2504.11268
120. Li et al. "AdaTT: Adaptive Task-to-Task Fusion Network". KDD 2023 (Meta). https://arxiv.org/abs/2304.04959
121. Su et al. "STEM: Shared and Task-specific EMbeddings" (AFTB gating). AAAI 2024. https://arxiv.org/abs/2308.13537 · https://github.com/LiangcaiSu/STEM
122. Wang et al. "HoME: Hierarchy of Multi-Gate Experts" (Kuaishou). KDD 2025. https://arxiv.org/abs/2408.05430
123. Chen et al. "Mod-Squad: MoE as Modular Multi-Task Learners". CVPR 2023. https://arxiv.org/pdf/2212.08066
124. "MTLoRA". CVPR 2024. https://arxiv.org/abs/2403.20320
125. Navon et al. "Learning the Pareto Front with Hypernetworks" (PHN). ICLR 2021. https://avivnavon.github.io/ParetoHN/
126. Dimitriadis et al. "Pareto Manifold Learning". ICML 2023. https://arxiv.org/abs/2210.09759
127. Dimitriadis et al. "PaLoRA: Pareto Low-Rank Adapters". ICLR 2025. https://arxiv.org/pdf/2407.08056
128. Ma et al. "Entire Space Multi-Task Model" (ESMM). SIGIR 2018. https://arxiv.org/pdf/1804.07931
129. Xi et al. "AITM: Adaptive Information Transfer Multi-task". KDD 2021. https://arxiv.org/abs/2105.08489
130. Yang et al. "CrossDistil: Cross-Task Knowledge Distillation in Multi-Task Recommendation". AAAI 2022. https://arxiv.org/abs/2202.09852
131. Lu et al. "Taskology: Utilizing Task Relations at Scale". CVPR 2020. https://arxiv.org/pdf/2005.07289
132. Li et al. "Learning Multiple Dense Prediction Tasks from Partially Annotated Data" (MTPSL). CVPR 2022. https://arxiv.org/pdf/2111.14893
133. "GradCraft". KDD 2024 (Kuaishou). https://arxiv.org/pdf/2407.19682
134. "Multi-Task Deep Recommender Systems: A Survey". 2023. https://arxiv.org/pdf/2302.03525
135. "Fine-grained task relatedness via data attribution". 2025. https://arxiv.org/html/2505.21438v1
136. "Model Merging in LLMs, MLLMs, and Beyond" survey. ACM CSUR. https://dl.acm.org/doi/10.1145/3787849
137. Yang et al. Foursquare TSMC-2014 dataset page. https://sites.google.com/site/yangdingqi/home/foursquare-dataset
138. Behrouz, Li, Deng, Zhong, Razaviyayn, Mirrokni. "Memory Caching: RNNs with Growing Memory" (GRM / Memory Soup / Sparse Selective Caching). arXiv:2602.24281, Feb 2026 (no code; OpenReview R3EJ2IjgOI). https://arxiv.org/abs/2602.24281 — user-flagged for layer-level gated-read exploration (`docs/studies/mtl_frontier/` R10).

## Datasets, protocol & surveys

95. Cho, Myers, Leskovec. "Friendship and Mobility: User Movement in Location-Based Social Networks" (Gowalla). KDD 2011. https://dl.acm.org/doi/10.1145/2020408.2020579
96. Luca, Barlacchi, Lepri, Pappalardo. "A Survey on Deep Learning for Human Mobility". ACM CSUR 2021. https://dl.acm.org/doi/10.1145/3485125
97. Luca et al. "Trajectory test-train overlap in next-location prediction datasets". Machine Learning 2023. https://link.springer.com/article/10.1007/s10994-023-06386-x
98. Islam et al. "A survey on deep learning based POI recommendations". Neurocomputing 2022. https://www.sciencedirect.com/science/article/abs/pii/S0925231221016106
99. Zhang et al. "A Survey on Point-of-Interest Recommendation: Models, Architectures, and Security". IEEE TKDE 2025. https://arxiv.org/abs/2410.02191
100. "A survey on GNN-based next POI recommendation for smart cities". J. Reliable Intelligent Environments 10, 2024. https://link.springer.com/article/10.1007/s40860-024-00233-z
