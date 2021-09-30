# Starting a PeARS pod with Wikipedia content

In this directory, we provide some code to start a PeARS pod on a particular topic, using Wikipedia as a guide to a) select the topic; b) gather initial content for the pod.


## Wikipedia category processing

We will first retrieve a number of topics from the Wikipedia category tree. We will use the [Wikimedia API](https://www.mediawiki.org/wiki/API:Main_page) for this:

    python3 get_categories.py

The code will write out all Wikipedia categories for which at least 200 documents exist, creating the file *wiki_categories.txt*. (It takes a minute or two, so be patient! In case you have difficulties running the code, you will find an already processed list in the data folder: *wiki_cats/preprocessed_wiki_categories.txt*.)

Once the initial extraction has run, we will cluster categories into larger 'meta-categories'. To do this, we will run:

     python3 map_wiki_cats.py --cats=wiki_cats/preprocessed_wiki_categories.txt

This will output a set of metacategories in the file *./wiki_cats/metacategories.txt*. Let us choose one of those metacategories, for instance, *genes on human chromosome*.


## Processing a meta-category

We are now ready to start our pod. Let's hash some documents for our metacategory.


### Extracting Wikipedia page titles

We will first get article titles for the meta-category of our choice. To do this, we will run:

    python3 get_category_pages.py

The script will request user input, i.e. the name of the meta-category you would like to process. For instance:

    Please enter a category name: genes on human chromosome

It will then output all Wikipedia categories found for your meta-category:

    ['Genes on human chromosome 19', 'Genes on human chromosome 9', 'Genes on human chromosome 20', 'Genes on human chromosome 16', 'Genes on human chromosome 5', 'Genes on human chromosome 4', 'Genes on human chromosome X', 'Genes on human chromosome 15', 'Genes on human chromosome 1', 'Genes on human chromosome 14', 'Genes on human chromosome 12', 'Genes on human chromosome', 'Genes on human chromosome 7', 'Genes on human chromosome 22', 'Genes on human chromosome 17', 'Genes on human chromosome 3', 'Genes on human chromosome 11', 'Genes on human chromosome 2', 'Genes on human chromosome 8', 'Genes on human chromosome 10', 'Genes on human chromosome 6']


Finally, for each subcategory of the metacategory, it will create a file *titles.txt* containing 20 Wikipedia page titles associated with that category, saved in the *data/* directory. (The limit of 20 pages is there in order not to overload Wikipedia at scraping stage. But you can of course change this limit, bearing in mind that you should be gentle on the Wikipedia API.)


### Scraping summaries from Wikipedia

The next step is to get actual content from Wikipedia, for all the pages we retrieved in our meta-category.

    python3 get_page_content.py

Again, the script will request our meta-category name.

    Please enter a category name: genes on human chromosome

This time, we will be retrieving page summaries from the Wikipedia API, saved in the appropriate *data/categories/* folder. For our example, you would see some Wikipedia summaries at the following location: *data/categories/genes_on_human_chromosome/Genes_on_human_chromosome/linear.txt*.

### Hashing the Wikipedia data

Finally, we produce hashes for our Wiki content. You should have a fly in the *fly/* directory, which we will use for hashing. We provide one for convenience, but you can make your own. Running the following will output document representations in the *hashes/* directory for the metacategory of our choice:

    python3 hash_pod.py --fly=fly/fly.m 
