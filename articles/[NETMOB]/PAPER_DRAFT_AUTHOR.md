## Title

[//]: # (WIP, let's define after the paper is kind done)

---

## 1. Introduction

LSBN datasest are a great oportunity to study the dynamics of social networks. One of the main challangers that
literatures bring is the prediction of next Point of interest (POI) that a user will visit. This problem can help to
improve the recommendation systems and to understand the mobility patterns of users, along to bring better insight for
future urban planing simulations. A side to this main task is the prediction of next category and next region that a
user will visit. Those tasks are important cause ussualy they are easier to decomposed and helps to understand and the
prediction of next POI.

In this paper we propose a new set of methodologies to predict the next category and next region of a POI that a user
will visit. The set starts with a new embedding engine that is able to learn the correlations between POIs, through the
buidl fo a graph that take considerations the category, time and distance; Then we propose a new a modern neural network
approach: Multitask learning, that is able to learn the next category and next region in a single model, and bring the
benefits to improve part of the result, due the shared knowledge between the tasks.

To support this study we count on two datasets, and an analyses over 5 states and one city, this diversity was on
purpose to evaluate the applicabilaty of our methids across small and huge datasets, and across different type of
datasets. The results show us that our embedding engine bring us substantial improvements over previous approaches,
while the multitask learning approach is able to improve the results of the category of next POI, keeping the results of
the next region. This finds are very interesting, on prove the effectiveness of this apporachs and bring us new tools
and analyzes over them so future works can make useful of them.


[//]: # (De We need to give background context about the check2hgi, that is based on a previus article ?)

---

## 2. Methods

[//]: # (Write about the gowalla and MassiveSteps datasets, show numbers of the used data of them[num. of cehckins, num. of users, max sequencies, sparsity, etc])

[//]: # (Write about the ETL process)

[//]: # (Write about the check2hgi methodollagy)
    [//]: # (Explain why traing all embedding then pass them to the MTL is not a dataleak)

[//]: # (Write about the MTL methodollagy, show the architecture of the model, and the loss function, and the training process)

---

## 3. Results

[//]: # (Comparisions of the Check2hgi using the images of the kNN-LOO, silhouette, separação de centroides, CKA, coerência de adjacência agains the hgi, and previus solutions)

[//]: # (Comparisions of the Check2hgi on State of the art solutions STL agains original solutions)

[//]: # (Explain what is STL[Single task learning], and compare the our STLs with the SOAT results)

[//]: # (Show the table result of the MTL for the 5 states[Gowalla] and the city[MassiveSteps], and compare with the best STL results, and explain the results)

---

## 4. Conclusion

[//] #

---