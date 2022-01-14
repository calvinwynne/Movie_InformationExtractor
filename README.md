# Movie Information-Extractor
Program that implements Natural Language Processing techniques and Information extraction algorithms to extract information from short movie description. Basic information like the director, producer, writers, cast and more can be extracted. 50 movie descriptions from the wikipedia corpus was taken as input along with their target gold-standard labels. This was then used as the input and to evaluate the performance of the model.

Part-of-speech (POS) tagging is a popular Natural Language Processing process which refers to categorizing words in a text (corpus) in correspondence with a particular part of speech, depending on the definition of the word and its context. Each document is broken down into sentences using the sentence tokenizer from nltk and then POS tagged using another built-in nltk function. This then is passed onto the chunking funciton which allows for named entity detection

Recognizing named entity is a specific kind of chunk extraction that uses entity tags along with chunk tags. Common entity tags include PERSON, LOCATION and ORGANIZATION. POS tagged sentences are parsed into chunk trees with normal chunking but the trees labels can be entity tags in place of chunk phrase tags. NLTK has already a pre-trained named entity chunker which can be used using ne_chunk() method in the nltk.chunk module. This method chunks a single sentence into a Tree
 
<img src="https://github.com/calvinwynne/Movie_InformationExtractor/blob/develop/movies/images/image3.png" width="450">


The program then tries to find verbs in pos-tagged and chunnked setences (with binary set to False). Upon finding verbs, it runs a search in both forward and backward direction to find all PERSON and ORGANIZATION entities. These entitites are then saved along with their disctances from the said detected verb. Depending on the best_match parameter, the function would either return the named entities closest to the verb or will return on all NEs with their respective distances. By no means does the closest NE to the verb mean its the required entity, but its better than having to choose the first entity in the list. There can be instances where there are 2 entities with similar distance from the verb, one before and the other after, in that case, I have chosen to go with the one after, this can be changed in the algorithm. This function is modular and generic, purpose built to work with any kind of sentences. Required verbs is alist of verbs that requires extracton. This in a way allows for fuzzy search, i.e the shorter the value in required verbs, the more results the model would retrieve. This obviously has a precision trade-off
as shown below:

Direc : Directed, Director, Direct (minght not be the intended value)
    
<img src="https://github.com/calvinwynne/Movie_InformationExtractor/blob/develop/movies/images/image1.gif" width="350">
