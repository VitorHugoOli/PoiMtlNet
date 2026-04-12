1. 


Enter in the branch copilot/create-improvement-plan-for-mtlnet(PR#6) and do a full /review of the code. Be    
  very critical, analysis on how we are managing the rest of the code and if this new changes are compatible    
  and if it make easy future expereiments and easy abalation studies   

----

5. Refactor codebase
   Prompt:
   Let's refactor this codebase. This is a project to study the application of the multitask learning in the context of
   classification unknown pois and prediction of next poi, the first task is static in way that recive a representation
   of a poi and try to classificaty it in one of the 7 categories; next in other hand is dynaic cause create a windoe of
   X embeddings and them try to predict what will be the next category of the embedding. So the main propose of this
   project is to contain the multitask learing algorithms, the neural network for the model and some support files
   arroun this system. Beyond that we keep some base lines and emebdding models that are related with this project but
   not the main focus. Not just that but this project contains severavl experiments, and maybe this is what make it more
   confuse, cause some parts of the main model have several different versions to test. Said that I would like you help
   to explore this project, search on the internet best pratices for this kind of proejct and them organize it and
   create a file structure and organization that allow me to test faster make change easier and obeserver better ne
   improvements. Feel free to use new toll new metodology of files and etc. Right now we are using the folder pipe to
   concentrated the main workflow there, but I dnt now if this is the best practice. Also fell free to change the code
   itself.

6. Start to write and search for the bracis [Paralell]
7. On the train pipe pass the len of the embeddings
8. Do we need to save the embedding model to validate in the mtlnet train ?
9. Improvement on the heads
10. Improvement on the MTLnET
    8.1 Let's do a analysis over the current /Users/vitor/Desktop/mestrado/ingred/src/model/mtlnet/mtl_poi.py. Get
    context in the documenations and md file about how the data is been mounted and what type os task we are trying
    to trains here and the flow to do this train. After that let's create a plan of how we could improve the model of
    the mt_poi, to get the best result, searhc on the literature best pratices and infromation os apporach that we
    are may missing. Some approach that I have resarch and I would like to try is the Mixture of expert, create a
    plan on this analysis on how we could add it on the mtl_poi. Be free to be radical and very critic. Notice that I
    have a previus anlysis in the mtlnet/UPDATE.md, you can use it as basis to do you searchs and suggestions. At the
    end Create a final UPDATE_CLAUDE.md in the /Users/vitor/Desktop/mestrado/ingred/src/model/mtlnet, with all the
    information gather.
11. Run the baselines
12. Execute flops and wall test to compare mtlnet
13. Analise de variancia de analise the hipotese estatistica
    --- Extras
13. Implemented new datasets
14. Top-k as metric

We use a implementation as based, but I would like you scrapy the article and the code related to this embedding to
ensure that our changes and waht we have is solid and is in vias with the orignal article propose. Also if not evalute
if worth to do the change or is expected. Be very critical and dig deep.

HGI (Hierarchical Graph Infomax)

Article: "Learning urban region representations with POIs and hierarchical graph
infomax"                                                                                                                                                                                                                                                                                                                                      
Authors: Weiming Huang, Daokun Zhang, Gengchen Mai, Xu Guo, Lizhen
Cui                                                                                                                                                                                                                                                                                                                                                         
Published: ISPRS Journal of Photogrammetry and Remote Sensing, vol. 196, pp. 134–145,
2023                                                                                                                                                                                                                                                                                                                                     
Code: https://github.com/RightBank/HGI                                                                                                                                                                                                                                                                                                                                                                                         
                                           