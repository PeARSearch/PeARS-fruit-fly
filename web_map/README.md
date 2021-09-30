# A map of knowledge

The end goal of PeARS Fruit Fly is to provide a highly efficient algorithm for users to hash Web documents. The produced hashes are grouped in topical [pods](https://pearsproject.org/faq.html#newpods). We can think of each pod as a the index of a mini search-engine, dedicated to a particular topic.

To kick off pod creation, we will provide already-seeded pods on a range of topics. We hope that more pods will be developed by end-users, expanding the knowledge range of the PeARS user base. In order to bring some order to this process, it would be useful to know the range of topics covered by Web content. This is a tall order, as it is impossible for us to know the entire Web consists of. However, we can provide some kind of 'topical grid' of human knowledge, which we will then use to order Internet content. We will create this grid from the English Wikipedia, as it is the most comprehensive knowledge repository at our disposal, publicly available and under Creative Commons license. 

This project directory contains two subdirectories:

* *metacat_eval*: code to evaluate the spatial configuration of Wikipedia categories after hashing with the Fruit Fly Algorithm. This folder is for development purposes only. It contains scripts to parse a Wikipedia snapshot, extract categories from it, cluster them into 'meta-categories', and link external Web documents to each category. The resulting dataset can then be used for hashing and evaluating the hashed space.
* *pod_starter*: code to allow end-users to create an initial pod for a given Wikipedia meta-category. This folder contains streamlined versions of the code in *metacat_val*, and tools to create fruit fly hashes of Wikipedia documents for their chosen topic, including a pretrained fly which gave high performance at development stage.


README files are available in both directories. For a more in-depth description of our mapping effort, visit the Wiki [here](https://github.com/PeARSearch/PeARS-fruit-fly/wiki/2.2.-Generate-document-representations-using-the-fruit-fly-algorithm).

