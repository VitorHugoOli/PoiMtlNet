Reviews
Review 1	
Appropriateness: 5: Certainly4: Probably3: Unsure2: Probably not1: Certainly not
Clarity: 5: Very clear4: Understandable by most readers3: Mostly understandable with some effort2: Difficult to follow1: Largely unclear or confusing
Originality: 5: Highly original4: Clearly creative and substantially different from prior work3: Meaningful extension of previous work2: Minor improvement over existing work1: Little or no originality
Soundness/Correctness: 5: The claims are convincingly supported4: Generally solid work, although there are some gaps3: Fairly reasonable work, but I am not entirely convinced to accept its conclusions and decisions2: Troublesome. The work should really have been done or evaluated differently1: Fatally flawed
Meaningful comparison: 5: Precise and complete given the space constraints4: Mostly solid bibliography and comparison, but there are some references missing3: Bibliography and comparison are somewhat helpful, but it could be hard for a reader to determine exactly how this work relates to previous work2: Only partial awareness and understanding of related work, or a flawed empirical comparison1: Little awareness of related work, or lacks necessary empirical comparison
Overall recommendation: 4: Accept3: Weak Accept2: Weak Reject1: Reject
Reviewer confidence: 3: High2: Moderate1: Low
Recommendation for best paper: 3: Definitely2: Maybe1: No
Does the paper fit the chosen track?: 1: Yes0: No
− Detailed comments:
This work proposes a model, called Check2HGI, for predicting points of interest (POIs) in location-based social networks. The model generates dynamic representation vectors at the check-in level rather than static ones based on physical location. The Check2HGI shows marginal gains in identifying the category of visited locations while scaling operational costs in extracting the target geographic region.

However, the manuscript presents methodological deficiencies that need to be addressed to validate its conclusions. The most serious error is a topological leak, in which the model's graph structure fully accesses the spatial transitions of the test data during pre-training, corrupting the symmetry of predictive induction. Furthermore, it is observed that the complex cross-attention network has limitations, operating with inferior performance compared to stochastic models such as Markov Chains.

Minor comment: there are some text problems (probably with latex), e.g., wrong line breaks. 
Review 2	
Appropriateness: 5: Certainly4: Probably3: Unsure2: Probably not1: Certainly not
Clarity: 5: Very clear4: Understandable by most readers3: Mostly understandable with some effort2: Difficult to follow1: Largely unclear or confusing
Originality: 5: Highly original4: Clearly creative and substantially different from prior work3: Meaningful extension of previous work2: Minor improvement over existing work1: Little or no originality
Soundness/Correctness: 5: The claims are convincingly supported4: Generally solid work, although there are some gaps3: Fairly reasonable work, but I am not entirely convinced to accept its conclusions and decisions2: Troublesome. The work should really have been done or evaluated differently1: Fatally flawed
Meaningful comparison: 5: Precise and complete given the space constraints4: Mostly solid bibliography and comparison, but there are some references missing3: Bibliography and comparison are somewhat helpful, but it could be hard for a reader to determine exactly how this work relates to previous work2: Only partial awareness and understanding of related work, or a flawed empirical comparison1: Little awareness of related work, or lacks necessary empirical comparison
Overall recommendation: 4: Accept3: Weak Accept2: Weak Reject1: Reject
Reviewer confidence: 3: High2: Moderate1: Low
Recommendation for best paper: 3: Definitely2: Maybe1: No
Does the paper fit the chosen track?: 1: Yes0: No
− Detailed comments:
The paper is difficult to follow, making it challenging to clearly understand the relationship between the presented literature, the actual contribution proposed by the authors, and how the experimental results support the main claims of the work. In addition, the novelty of the paper is not sufficiently well positioned within the Related Work section. The manuscript does not clearly explain how the proposed approach differs from or improves upon existing methods, which weakens the perception of originality and contribution. The Results section is also difficult to interpret. It is not clear whether the proposed method consistently outperforms competing approaches, and the discussion relies heavily on dense numerical tables without providing sufficient guidance or interpretation for the reader. As a result, the experimental analysis does not effectively communicate the practical significance or strengths of the proposed method.

Overall, the paper would benefit from a clearer narrative structure, a stronger positioning of the novelty with respect to prior work, and a more interpretative discussion of the experimental results. 
Review 3	
Appropriateness: 5: Certainly4: Probably3: Unsure2: Probably not1: Certainly not
Clarity: 5: Very clear4: Understandable by most readers3: Mostly understandable with some effort2: Difficult to follow1: Largely unclear or confusing
Originality: 5: Highly original4: Clearly creative and substantially different from prior work3: Meaningful extension of previous work2: Minor improvement over existing work1: Little or no originality
Soundness/Correctness: 5: The claims are convincingly supported4: Generally solid work, although there are some gaps3: Fairly reasonable work, but I am not entirely convinced to accept its conclusions and decisions2: Troublesome. The work should really have been done or evaluated differently1: Fatally flawed
Meaningful comparison: 5: Precise and complete given the space constraints4: Mostly solid bibliography and comparison, but there are some references missing3: Bibliography and comparison are somewhat helpful, but it could be hard for a reader to determine exactly how this work relates to previous work2: Only partial awareness and understanding of related work, or a flawed empirical comparison1: Little awareness of related work, or lacks necessary empirical comparison
Overall recommendation: 4: Accept3: Weak Accept2: Weak Reject1: Reject
Reviewer confidence: 3: High2: Moderate1: Low
Recommendation for best paper: 3: Definitely2: Maybe1: No
Does the paper fit the chosen track?: 1: Yes0: No
− Detailed comments:
The paper presents a model for point-of-interest vector representation and evaluates its performance in next-location and POI category classification tasks. The work extends the HGI model to add an extra level in the hierarchy related to "check-in", or temporally-relevant customer visit. 
My principal criticism is that the presentation of both methods and results is very difficult to follow. There should be more space dedicated to the background concepts (DGI and HGI) and a brief description of the cited competing approaches. Also, I wasn't able to convincingly grasp the structure of the input graph or the model network from the explanations, and though the pitch of the results is appealing, I can only be sure that the tradeoff between tasks is inevitable if I can rule out structural bottlenecks, which require a clearer picture of the different model configurations. In short, the paper need a rework in its presentation, rather than its content. 