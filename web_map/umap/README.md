# A map of knowledge - dimensionality reduced implementation

This folder contains code to process the whole of Wikipedia in any language of interest, to generate a 'topical map' of encyclopedic knowledge. This is a specific implementation of the idea of a 'knowledge map', explained [here](https://github.com/PeARSearch/PeARS-fruit-fly/tree/main/web_map). Implementation specifics are as follows:

* the code can be run for any language with a Wikipedia;
* it implements dimensionality reduction prior to feeding document representations to the fruit fly;
* the reduced representations are turned into binary vectors via the fly;
* evaluation is performed using *precision @ k*, a measure that computes whether the *k* nearest neighbours of a target document belong to the same class as the target;
* classes for evaluation are provided by prior clustering over the reduced representations (but prior to binarization).


