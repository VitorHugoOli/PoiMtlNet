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


We had a problem on the dataset where some of the labels was wrong this generate imcompatible
  resutls to compare to the other baselines. I will rerun all the embedding. So let's take this
  oprtunity and based on your finds runing the full_abalation, issues listed, objectives to test,
  claims that we hypothisais about the finds and implementations on this project. Let's create a
  new full abalation study, we will have the alabama and arizona as fast executions and florida 
  for slow but huge valitations(also if necessary more data for diferrent comparasion with     
  baseline or something like that just aky). I belive having more datasets will incrise your     
  confibility on the results. Also let's reviseted the paper findings(/Volumes/Vitor's           
  SSD/ingred/docs/PAPER_FINDINGS.md) and validation them; not only them but also let's create a  
  list of new the claims that we expected and those ones that make sense considering to studies  
  based on what we are doing. I belive a good approach would have a doc file with all them       
  claims/hypoteses and for each of them we run a test and point to this documentation and write  
  the justification. But before all this and yeat in the full abalation study let's start runing 
  a full comparision with the 5 arch x 20 otmizers, during this abalation we can at same time    
  evalute some parts of the claims like the `radient-surgery optimizers that resolve the         
  scale-imbalance-induced` and other claims that you judge pertnates to evalute in this phase; so
  not just create a abalation execution but also take tthe advantege and do your studies at      
  tohgeter for what make sense. After this first level of abalations and studies we could focus  
  on the heads, and do a real and deep analys on the 9 category and 10 next, on this abalation we
  could also sutdy the benefits of mtls over the single standalone training, and create          
  hipotestes over why it improves or not improves, also keep using this to study the claims and  
  hypoteses that fits on this moments. Also after defines the arch, optmizers and heads of study 
  I belive would be great to run some of them in the three main embeddings DGI,HGI and Fusion, so
  we can have comparations and make easier to anwer some question about if the arch and optimize
  is realy doing any diference(May could be good to run all of the three in the cbic arch to     
  have as base on how we improve from cbic and have more argunamtation). At the end after choise 
  the best mtl model, optimizer and heads let's focus on a abalations studie for the            
  hyperparameters of the mtl model itself. Finally after all this if there is claims and         
  hypotests that needs other kind of executations and vlaidation, do them.     Clould be good to 
  save all the results along this test and execution so in future if we have other claims and    
  hypotests we could back on this data to valitad and ansewr some question.     Organize all my  
  ideias, do your analyzes and think on what can be run in parallel on the same computer on in     
  other computers(I can run on them and bring the result to here and save on the set folder);      
  Also think on the before of the abalation, executions and tests if we must prepare or create     
  some tools,mechanism or code to make this study easy, trackeble and observable. Save this        
  inital plan on a folder call studies on the docs/folder make it easy for humans and agents       
  localize and understand on each step we are of the tests. Also since we can have a lot of       
  parallel execution and executions that can stop of broken in the middle let's create a system   
  of a opus cordenator where it will controll the current step and the current test and how the    
  test will be done also validet if the test are in the correct place and if the data are correct  
  and integraty, if some suspectios appear it should stop warning and request new tests, also      
  this cordinator(or other agent, what you judges that fits better) also must think criticaly      
  about the result that it is reciving and if new test claims can be made or also we could         
  theorisys new hypoteses. At the end all of this must be sintizes in a new doc paper_findinds.
