import sys.process._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Map
import java.io.File 
import java.io.PrintWriter
import org.clulab.processors.corenlp.CoreNLPProcessor
import org.clulab.processors.shallownlp.ShallowNLPProcessor
import org.clulab.processors.{Document, Processor}
import org.clulab.struct.DirectedGraphEdgeIterator
import org.clulab.processors.fastnlp.FastNLPProcessor

/*
 * Preprocesses corpus:
 *   1. Extracts depencency trees from each sentence of corpus
 *   2. Extracts paths from dependency tree and applies the following filtering:
 *      1. Both slot filler words must be named entities or nouns
 *      2. The frequency of the path must be above a threshold
 *   3. Saves the data in the form several tsv files to be used by the further programs
 *
 *  Run this file using the following command:
 *  mvn  scala:run  -DmainClass=Preprocess_Part1  "-DaddArgs=arg1|arg2"
 *  In BERT mode, you need to provide two commmand-line arguments:
 *  arg1: virtual environment path for running create_BERTtoken_mappings.py
 *  arg2: BERT model name that will be passed to create_BERTtoken_mappings.py
 */

object Preprocess_Part1 {
    def main(args:Array[String]) {
    
        // determines whether preprocessing is done for BERT or DIRT
        val BERT_mode = true
        // maximum number of input tokens BERT can accept is 512 
        val BERT_MAX_TOKEN_INDEX = 511
        
        val files_dir = "src/main/scala/mahdi/data/"
        val preprocess_dir = "./src/main/scala/mahdi/preprocess/"
        val call_create_BERTtoken_mappings_filename = "call_create_BERTtoken_mappings"
    
        /* list of named entity classes used for filtering paths */
        val namedEntityClasses = List("PERSON", "LOCATION", "ORGANIZATION", "MISC", "MONEY", "NUMBER", "ORDINAL", "PERCENT", "DATE", "TIME",
                                      "DURATION", "SET", "EMAIL", "URL", "CITY", "STATE_OR_PROVINCE", "COUNTRY", "NATIONALITY", "RELIGION",
                                      "TITLE", "IDEOLOGY", "CRIMINAL_CHARGE", "CAUSE_OF_DEATH", "HANDLE")
        
        /* list of part-of-speech tags for nouns */
        val nounPosTags = List("CD", "NN", "NNS", "NNP", "NNPS", "PRP", "WP", "WDT", "WP$")
        
        /* paths_db is the container for the extracted paths along with their features and sentences
        *  paths_db = { path , (slotX_words_freq , slotY_words_freq , sentences_info) }
        *  slotX_words_freq = {word , freq}
        *  slotY_words_freq = {word , freq} 
        *  sentences_info = [(slotX_start_char_index, slotX_end_char_index, slotX_word, slotY_start_char_index, slotY_end_char_index, slotY_word, sentence_id)]
        */
        var paths_db: Map[ArrayBuffer[(String, String, String, String)] , 
                          (Map[String,Int] , Map[String,Int] , ArrayBuffer[(Int,Int,String,Int,Int,String,Int)])]  = Map()
        var filtered_paths_db: Map[ArrayBuffer[(String, String, String, String)] , 
                                   (Map[String,Int] , Map[String,Int] , ArrayBuffer[(Int,Int,String,Int,Int,String,Int)])]  = Map()

        /* slotX_features_db and slotY_features_db = {word , List[path_ids]}
           The dictionaries that for each word store the list of path ids 
           of the paths that the word fills a slot of (slotX and slotY).
           These dictionaries are created to speed up the calculation of 
           similary scores later in the pipeline */
        var slotX_features_db: Map[String , List[Int]] = Map()
        var slotY_features_db: Map[String , List[Int]] = Map()
        
        /* dictionary for storing mappings between sentenceids and sentences.
           sentence_to_sentenceid = {sentence : sentence_id} */
        var sentence_to_sentenceid: Map[String , Int] = Map()

        // create FastNLP Processor
        var proc:Processor = new FastNLPProcessor()
        
        // read the contents of corpus
        val corpus = scala.io.Source.fromFile(files_dir + "corpus", "UTF-8").mkString
        
        // run the FastNLP Processer on the corpus
        var doc = proc.annotate(corpus)
        
        val totalSentences: Double = doc.sentences.length
    
        println("\n")
        println("Number of total sentences: " + Math.round(totalSentences))
        println()
        println("Extracting sentences...")
        
        // iterate over each sentence in the corpus
        var sentenceCount = 0
        var progressMileStone = 0.05
        var new_sen_id = 0
        var sen_id = -1
        for (sentence <- doc.sentences) {
            
            // update progress indicator
            if ((sentenceCount/totalSentences) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
            
            // update sentence_to_sentenceid
            val sen = sentence.words.mkString(" ")
            if (!sentence_to_sentenceid.contains(sen))
            {
                sen_id = new_sen_id
                sentence_to_sentenceid(sen) = sen_id
                new_sen_id += 1
            }
            
            sentenceCount += 1
        }
        println("100%\n")
        
        // write the mapping of sentenceids to sentences to a file (sentenceids_to_sentences.tsv)
        println("Writing 'sentenceids_to_sentences.tsv' to disk...")
        val sentenceids_to_sentences_file = new File(files_dir + "sentenceids_to_sentences.tsv")
        val sentenceids_to_sentences_writer = new PrintWriter(sentenceids_to_sentences_file, "UTF-8")
        
        var sc = 0
        progressMileStone = 0.05
        val dlm = "\t"
        val totalUniqueSentences: Double = sentence_to_sentenceid.size
        for ((sentence , sentence_id) <- sentence_to_sentenceid)
        {
            if ((sc/totalUniqueSentences) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
        
            sentenceids_to_sentences_writer.write(sentence_id.toString +
                                                  dlm +
                                                  "\"" + process_string(sentence) + "\"" +
                                                  "\n")
                                                  
            sc += 1
        }
        sentenceids_to_sentences_writer.close()
        println("100%\n")

        var sentence_chars_to_BERTtokens_indexes: Map[Int , Map[Int,Int]] = Map()
        var chars_to_BERTtokens_indexes: Map[Int,Int] = Map()

        // in BERT mode, fill sentence_chars_to_BERTtokens_indexes dictionary 
        // from the corresponding file
        if (BERT_mode)
        {
            // create "sentence_chars_to_BERTtokens_indexes.tsv"
            val virtual_environment_path = args(0)
            val BERT_model_name = args(1)
            preprocess_dir + call_create_BERTtoken_mappings_filename + " " + virtual_environment_path + " " + BERT_model_name !
            
            // load "sentence_chars_to_BERTtokens_indexes.tsv"
            val sentence_chars_to_BERTtokens_indexes_file = files_dir + "sentence_chars_to_BERTtokens_indexes.tsv"
            val sctbi_bufferedSource = scala.io.Source.fromFile(sentence_chars_to_BERTtokens_indexes_file, "UTF-8")

            for (line <- sctbi_bufferedSource.getLines)
            {
                chars_to_BERTtokens_indexes = Map()
            
                var fields = line.split("\t")
                var sen_id = fields(0).toInt
                
                var col2 = fields(1)
                col2 = col2.substring(1 , col2.length-1)  //trim '{' and '}'
                
                var col2_fields = col2.split(",")
                for (charidx_tokenidx_str <- col2_fields)
                {
                    var charidx_and_tokenidx = charidx_tokenidx_str.split(":")
                    var char_idx = charidx_and_tokenidx(0).toInt
                    var BERT_token_idx = charidx_and_tokenidx(1).toInt
                    chars_to_BERTtokens_indexes(char_idx) = BERT_token_idx
                }
                sentence_chars_to_BERTtokens_indexes(sen_id) = chars_to_BERTtokens_indexes
            }
            sctbi_bufferedSource.close
        }
        
        println("\n")
        println("Extracting paths...")
        
        // iterate over each sentence in the corpus and extract paths
        sentenceCount = 0
        progressMileStone = 0.05
        sen_id = -1
        for (sentence <- doc.sentences) {
            
            // update progress indicator
            if ((sentenceCount/totalSentences) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
            
            // get lemmas for the sentence
            val lemmas = sentence.lemmas match {
                            case Some(i) => i
                            case None => Array[String]()}
            
            // get pos tags for the sentence
            val tags = sentence.tags match {
                            case Some(i) => i
                            case None => Array[String]()}
            
            // get named entities for the sentence
            val entities = sentence.entities match {
                            case Some(i) => i
                            case None => Array[String]()}
            
            // get the lemmatized words of the sentence
            val tokens = sentence.words.zipWithIndex.map{ case (element, index) => lemmas(index) }
            
            // get sentence id of the sentence
            val sen = sentence.words.mkString(" ")
            sen_id = sentence_to_sentenceid(sen)
            
            // get the dependency tree for the sentence
            val deps = sentence.dependencies.get
            
            var multi_word_group = create_multi_word_groups(entities, namedEntityClasses)
            
            // store the observed unprocessed paths of the sentence
            var observed_unprocessed_paths : ArrayBuffer[Seq[(Int, Int, String, String)]] = ArrayBuffer()
            
            // if the sentence is incompatible with its corresponding data in 
            // 'sentence_chars_to_BERTtokens_indexes.tsv', discard the sentence. 
            // Otherwise, proceed to extract paths.
            var discard_sentence = false
            if (BERT_mode)
                if ((sen.length-1) != (sentence_chars_to_BERTtokens_indexes(sen_id).keysIterator.max))
                    discard_sentence = true
            
            if (!discard_sentence)
            {
              // extract paths
              for (start <- 0 until tokens.length-1)
                for (end <- 0 until tokens.length-1)
                {
                    if (start != end)
                    {
                        // get the raw path
                        val paths = deps.shortestPathEdges(start, end, ignoreDirection = true)
                        var unprocessed_path = paths.head
                        
                        /* on both endpoints of the path, remove all depencency relations that are 
                           internal parts of multi-words. For example, in the following path:
                           John Smith went to New York
                           John<-compound<-Smith<-nsubj<-go->nmod_to->York->compound->New
                           remove  John<-compound<-Smith  and  York->compound->New
                           so we'll have:
                           Smith<-nsubj<-go->nmod_to->York
                         */
                        var (trimmed_unprocessed_path , slotX_start_idx , slotX_end_idx , slotY_start_idx , slotY_end_idx) 
                            = trim_multiword_slotfillers(unprocessed_path, multi_word_group)
                        unprocessed_path = trimmed_unprocessed_path

                        var slotX_start_char_idx = -1 
                        var slotX_end_char_idx = -1
                        var slotY_start_char_idx = -1 
                        var slotY_end_char_idx = -1
                        
                        // only keep paths that their slot-fillers are named entities or nouns.
                        // also, in BERT mode, only accept paths with 
                        // BERT tokens indexes <= BERT_MAX_TOKEN_INDEX-1.
                        var keep = false
                        if (unprocessed_path.length > 0)
                        {
                            var slotX_char_span = get_char_span_in_sentence(slotX_start_idx , slotX_end_idx , sentence.words)
                            slotX_start_char_idx = slotX_char_span._1
                            slotX_end_char_idx = slotX_char_span._2

                            var slotY_char_span = get_char_span_in_sentence(slotY_start_idx , slotY_end_idx , sentence.words)
                            slotY_start_char_idx = slotY_char_span._1
                            slotY_end_char_idx = slotY_char_span._2
                        
                            if (BERT_mode)
                                keep = is_valid_BERT_token_index(slotX_start_char_idx , slotX_end_char_idx ,
                                                                 slotY_start_char_idx , slotY_end_char_idx ,
                                                                 sentence_chars_to_BERTtokens_indexes , 
                                                                 sen_id , 
                                                                 BERT_MAX_TOKEN_INDEX) &&
                                       keep_path(namedEntityClasses, entities, nounPosTags, tags, slotX_start_idx, slotY_start_idx)
                            else
                                keep = keep_path(namedEntityClasses, entities, nounPosTags, tags, slotX_start_idx, slotY_start_idx)
                        }
                        
                        // because of multi-word slot-fillers, there may be redundant
                        // unprocessed paths. make sure to process them only once.
                        // also, only keep the paths we want to keep.
                        if ((!observed_unprocessed_paths.contains(unprocessed_path)) && keep)
                        {
                            // add unprocessed_path to the list of observed paths
                            observed_unprocessed_paths.append(unprocessed_path)
                            
                            var slotX_word = tokens.slice(slotX_start_idx , slotX_end_idx).mkString(" ")
                            var slotY_word = tokens.slice(slotY_start_idx , slotY_end_idx).mkString(" ")
                        
                            // process the path; replace indexes with tokens, strip 
                            // slot-filler words, and store the path in a new data structure 
                            var path = processPath(unprocessed_path, tokens)

                            // if the path is not new, update paths_db accordingly
                            if (paths_db.contains(path))
                            {
                                var (slotX , slotY , sentences_info) = paths_db(path)
                            
                                if (slotX.contains(slotX_word))
                                    slotX(slotX_word) += 1
                                else
                                    slotX(slotX_word) = 1
                                
                                if (slotY.contains(slotY_word))
                                    slotY(slotY_word) += 1
                                else
                                    slotY(slotY_word) = 1
                                
                                sentences_info.append((slotX_start_char_idx, slotX_end_char_idx, slotX_word, 
                                                       slotY_start_char_idx, slotY_end_char_idx, slotY_word,
                                                       sen_id)) 
                            }
                        
                            // if the path is new, add it to paths_db
                            else
                            {
                                paths_db(path) = (Map(slotX_word -> 1) , 
                                                  Map(slotY_word -> 1) , 
                                                  ArrayBuffer((slotX_start_char_idx, slotX_end_char_idx, slotX_word, 
                                                               slotY_start_char_idx, slotY_end_char_idx, slotY_word,
                                                               sen_id)))
                            }
                        }
                    }
                }
            
              sentenceCount += 1
            }
        }
        
        println("100%\n")
        
        // trying to release the memory for the processor and 
        // its annotated object. Hopefully the garbage collector
        // will release their memory.
        doc = null
        proc = null
        
        var totalPaths: Double = paths_db.size
        println("Number of extracted paths: " + Math.round(totalPaths))
        
        println()
        println("Creating paths frequency database...")
        
        // create a list that stores the frequency of each path
        var paths_freq: List[(ArrayBuffer[(String, String, String, String)] , Int)] = List()
        var c = 0
        progressMileStone = 0.05
        for((path, (slotX,_,_)) <- paths_db)
        {
            if ((c/totalPaths) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
        
            var freq = 0
            for( (_,f) <- slotX )
                freq += f
        
            paths_freq = (path,freq) :: paths_freq
            
            c += 1
        }
        
        println("100%\n")
        println("Sorting the paths frequency database...\n")
        
        // sort the list that stores the frequency of each path
        paths_freq = paths_freq.sortBy(_._2)
        
        // write the list to a file (paths_freq.tsv)
        println("Writing the paths frequency database to disk...")
        c = 0
        progressMileStone = 0.05
        val paths_freq_file = new File(files_dir + "paths_freq.tsv")
        val paths_freq_writer = new PrintWriter(paths_freq_file, "UTF-8")
        for (i <- (paths_freq.length-1) to 0 by -1)
        {
            if ((c/totalPaths) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
        
            var path = paths_freq(i)._1
            var frequency = paths_freq(i)._2
            val delim = "\t"
            paths_freq_writer.write(get_path_textual_string(path) +
                                    delim +
                                    frequency +
                                    "\n")
                                    
            c += 1
        }
        paths_freq_writer.close()
        println("100%\n")
        
        // paths that have frequency below this threshold will be filtered
        val freq_threshold = 10
        
        println("Applying final paths filtering using frequency threshold of " + freq_threshold + "...")
        
        // find out how many paths have frequency equal or above the threshold
        var continue = true
        var totalRemainingPaths = 0
        var i = paths_freq.length-1
        while(continue)
        {
            var frequency = paths_freq(i)._2
            if (frequency < freq_threshold)
                continue = false
            else
                i -= 1
            
            if (i == -1) continue = false
        }
        totalRemainingPaths = paths_freq.length - i - 1
        
        // filter the paths with frequency below the threshold
        val total_rem_paths_double: Double = totalRemainingPaths
        c = 0
        progressMileStone = 0.05
        for (i <- (paths_freq.length-1) to (paths_freq.length-totalRemainingPaths) by -1)
        {
            if ((c/total_rem_paths_double) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
            
            var path = paths_freq(i)._1
            filtered_paths_db(path) = paths_db(path)
            
            c += 1
        }
        paths_db = Map()
        paths_db = filtered_paths_db
        println("100%\n")
        
        totalPaths = paths_db.size.toDouble
        println("Number of total paths after final filtering: " + Math.round(totalPaths))
        println()
        val delim = "\t"
        
        /* write three files:
         *   pathids_to_paths.tsv: a mapping from path ids to paths
         *   paths_features.tsv  : a mappgin from path ids to features of each slot of the paths
         *   paths_sentences.tsv : a mapping from path ids to the ids of the sentences that have the paths 
         */
        println("Writing 'pathids_to_paths.tsv', 'paths_features.tsv', and 'paths_sentences.tsv' to disk...")

        val pathids_to_paths_file = new File(files_dir + "pathids_to_paths.tsv")
        val pathids_to_paths_writer = new PrintWriter(pathids_to_paths_file, "UTF-8")
        
        val paths_features_file = new File(files_dir + "paths_features.tsv")
        val paths_features_writer = new PrintWriter(paths_features_file, "UTF-8")
        
        val paths_sentences_file = new File(files_dir + "paths_sentences.tsv")
        val paths_sentences_writer = new PrintWriter(paths_sentences_file, "UTF-8")
        
        var paths_to_pathids_db: Map[ArrayBuffer[(String, String, String, String)] , Int] = Map()
        
        var counter = 0
        progressMileStone = 0.05
        for((path, (slotX,slotY,sentences_info)) <- paths_db)
        {
            if ((counter/totalPaths) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
        
            val delim = "\t"
            
            pathids_to_paths_writer.write(counter.toString +
                                          delim +
                                          get_path_datastructure_string(path) +
                                          "\n")

            paths_to_pathids_db(path) = counter
            
            paths_features_writer.write(counter.toString +
                                        delim + 
                                        get_pathslot_datastructure_string(slotX) +
                                        delim +
                                        get_pathslot_datastructure_string(slotY) +
                                        "\n")
            
            paths_sentences_writer.write(counter.toString +
                                         delim +
                                         get_pathsent_datastructure_string(sentences_info) +
                                         "\n")
            counter += 1
        }
        pathids_to_paths_writer.close()
        paths_features_writer.close()
        paths_sentences_writer.close()
        println("100%\n")

        println("Creating slotX and slotY features to paths databases...")
        
        /* create two dictionaries slotX_features_db and slotY_features_db that
         * each contain a mapping from words to the lists of path ids of the paths
         * that the words have been used in as a slot-filler. One for slotX and one 
         * for slotY. */
        c = 0
        progressMileStone = 0.05
        for((path, (slotX,slotY,_)) <- paths_db)
        {
            if ((c/totalPaths) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
        
            var path_id = paths_to_pathids_db(path)
        
            for ((word,f) <- slotX)
            {
                if (slotX_features_db.contains(word))
                    slotX_features_db(word) = slotX_features_db(word) :+ path_id
                else
                    slotX_features_db(word) = List(path_id)
            }
            
            for ((word,f) <- slotY)
            {
                if (slotY_features_db.contains(word))
                    slotY_features_db(word) = slotY_features_db(word) :+ path_id
                else
                    slotY_features_db(word) = List(path_id)
            }
            
            c += 1
        }
        
        println("100%\n")
        
        val totalSlotXFeatures: Double = slotX_features_db.size
        val totalSlotYFeatures: Double = slotY_features_db.size
        println("Number of total slotX features (words): " + Math.round(totalSlotXFeatures))
        println("Number of total slotY features (words): " + Math.round(totalSlotYFeatures))
        println()
        
        println("Writing 'xfeatures_paths.tsv' to disk...")

        // write the previously created dictionaries to files 
        // xfeatures_paths.tsv and yfeatures_paths.tsv
        val Xfeatures_paths_file = new File(files_dir + "xfeatures_paths.tsv")
        val Xfeatures_paths_writer = new PrintWriter(Xfeatures_paths_file, "UTF-8")
        
        c = 0
        progressMileStone = 0.05
        for ((word,path_ids) <- slotX_features_db)
        {
            if ((c/totalSlotXFeatures) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
        
            val delim = "\t"
            
            var line = "\"" + process_string(word) + "\"" + delim + "["
            for (path_id <- path_ids)
            {
                line += path_id + ","
            }
            line = line.substring(0 , line.length-1) + "]\n"
            
            Xfeatures_paths_writer.write(line)
            
            c += 1
        }
        Xfeatures_paths_writer.close()
        println("100%\n")
        
        println("Writing 'yfeatures_paths.tsv' to disk...")
        
        val Yfeatures_paths_file = new File(files_dir + "yfeatures_paths.tsv")
        val Yfeatures_paths_writer = new PrintWriter(Yfeatures_paths_file, "UTF-8")
        
        c = 0
        progressMileStone = 0.05
        for ((word,path_ids) <- slotY_features_db)
        {
            if ((c/totalSlotYFeatures) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
        
            val delim = "\t"
            
            var line = "\"" + process_string(word) + "\"" + delim + "["
            for (path_id <- path_ids)
            {
                line += path_id + ","
            }
            line = line.substring(0 , line.length-1) + "]\n"
            
            Yfeatures_paths_writer.write(line)
            
            c += 1
        }
        Yfeatures_paths_writer.close()
        println("100%\n")
    }
    
    
    /* processes a string by replacing a backslash with 
      two backslashes and adds a backslash before a quotation 
      mark. This is necessary for writing strings to a file 
      that will be later read by another program.
    */
    def process_string(s: String) : String =
    {
        var ret_val = s.replace("\\" , "\\\\")
        ret_val = ret_val.replace("\"" , "\\\"")
        return ret_val
    }
    
    
    def reverse_process_string(s: String) : String =
    {
        var ret_val = s.replace("\\\\" , "\\")
        ret_val = ret_val.replace("\\\"" , "\"")
        return ret_val
    }
    
    /* creates a text string for a slot data (which is a dictionary).
       This text is meant to be written to a file and later be read 
       by the Python program that uses this data. 
     */
    def get_pathslot_datastructure_string(slot:Map[String,Int]) : String =
    {
        var ret_val = ""
        ret_val += "{"
        for((word,freq) <- slot)
        {
            ret_val += "\"" + process_string(word) + "\":" + freq + "," 
        }
        ret_val = ret_val.substring(0 , ret_val.length-1) + "}"
        return ret_val
    }
    
    /* checks whether should keep the given path or filter it.
       slot fillers must be named entities or nouns.
     */
    def keep_path(namedEntityClasses:List[String], entities:Array[String], 
                  nounPosTags:List[String], tags:Array[String],
                  slotX_idx: Int, slotY_idx: Int) : Boolean =
    {
        var ret_val = false
        
        // Slot fillers must be named entities or nouns
        if ((namedEntityClasses.contains(entities(slotX_idx)) || nounPosTags.contains(tags(slotX_idx))) &&
            (namedEntityClasses.contains(entities(slotY_idx)) || nounPosTags.contains(tags(slotY_idx))))

            ret_val = true
    
        return ret_val
    }

    /* process the given raw path:
         - replace token indexes with tokens
         - strip the slot-filler words from the path and return them in separate variables
         - return the indexes of the slot-filler words
         - return the path in a new data structure
     */
    def processPath(unprocessed_path:Seq[(Int, Int, String, String)], tokens:Array[String]) : 
        ArrayBuffer[(String, String, String, String)] = 
    {   
        val slot_filler_string = "_"
        var processed_path = ArrayBuffer[(String, String, String, String)]()
        
        for (t <- unprocessed_path)
        {
            var new_t = (tokens(t._1) , tokens(t._2) , t._3 , t._4)
            processed_path += new_t
        }
        
        if (processed_path(0)._4 == "<")
        {
            processed_path(0) = processed_path(0).copy(_2 = slot_filler_string)
        }
        else
        {
            processed_path(0) = processed_path(0).copy(_1 = slot_filler_string)
        }
        
        val last_idx = processed_path.length - 1
        if (processed_path(last_idx)._4 == ">")
        {
            processed_path(last_idx) = processed_path(last_idx).copy(_2 = slot_filler_string)
        }
        else
        {
            processed_path(last_idx) = processed_path(last_idx).copy(_1 = slot_filler_string)
        }
        
        return processed_path
    }

    /* print a textual representation for a raw (unprocessed) path. 
     */
    def print_unprocessedpath_textual(path:Seq[(Int, Int, String, String)], tokens:Array[String]) 
    {
        var last_printed_element = ""
        var first_element_to_print = ""
        for (t <- path)
            if (t._4 == ">")
            {
                first_element_to_print = tokens(t._1)
                if (first_element_to_print == last_printed_element)
                    first_element_to_print = ""
                        
                print(first_element_to_print + "->" + t._3 + "->" + tokens(t._2))
                last_printed_element = tokens(t._2)
            }
            else
            {
                first_element_to_print = tokens(t._2)
                if (first_element_to_print == last_printed_element)
                    first_element_to_print = ""
                    
                print(first_element_to_print + "<-" + t._3 + "<-" + tokens(t._1))
                last_printed_element = tokens(t._1)
            }
        print("\n")
    }

    /* get a textual representation for a processed path. 
     */
    def get_path_textual_string(processed_path:ArrayBuffer[(String, String, String, String)]) : String = 
    {
        var ret_val = ""
        var last_printed_element = ""
        var first_element_to_print = ""
        for (t <- processed_path)
            if (t._4 == ">")
            {
                first_element_to_print = t._1
                if (first_element_to_print == last_printed_element)
                    first_element_to_print = ""
                        
                ret_val += first_element_to_print + "->" + t._3 + "->" + t._2
                last_printed_element = t._2
            }
            else
            {
                first_element_to_print = t._2
                if (first_element_to_print == last_printed_element)
                    first_element_to_print = ""
                    
                ret_val += first_element_to_print + "<-" + t._3 + "<-" + t._1
                last_printed_element = t._1
            }
        return ret_val
    }
    
    /* creates a text string for a path. This text is meant to be
       written to a file and later be read by the Python program 
       that uses this data. 
     */    
    def get_path_datastructure_string(processed_path:ArrayBuffer[(String, String, String, String)]) : String =
    {
        var counter = 1
        var ret_val = ""
        
        ret_val += "["
        for (t <- processed_path)
        {
            ret_val += "(" +
                       "\"" + process_string(t._1) + "\"," +
                       "\"" + process_string(t._2) + "\"," +
                       "\"" + process_string(t._3) + "\"," +
                       "\"" + process_string(t._4) + "\""  +
                       ")"
            
            if (counter < processed_path.length)
                ret_val += ","
                
            counter +=1
        }
        
        ret_val += "]"
        return ret_val
    }
    
    /* creates a text string for a path sentence info (third element of 
       values of paths_db dictionary). This text is meant to be written 
       to a file and later be read by the Python program that uses this data. 
     */    
    def get_pathsent_datastructure_string(sentences_info: ArrayBuffer[(Int,Int,String,Int,Int,String,Int)]) : String =
    {
        var counter = 1
        var ret_val = ""
        
        ret_val += "["
        for (t <- sentences_info)
        {
            ret_val += "(" +
                       t._1.toString + "," +
                       t._2.toString + "," +
                       "\"" + process_string(t._3) + "\"," +
                       t._4.toString + "," +
                       t._5.toString + "," +
                       "\"" + process_string(t._6) + "\"," +
                       t._7.toString + ")"
            
            if (counter < sentences_info.length)
                ret_val += ","
                
            counter +=1
        }
        
        ret_val += "]"
        return ret_val    
    }

    
    def trim_multiword_slotfillers(unprocessed_path:Seq[(Int, Int, String, String)] , multi_word_group:Map[Int , List[Int]])
        : (Seq[(Int, Int, String, String)] , Int , Int , Int , Int) =
    {    
        var i = 0
        var last_idx = unprocessed_path.length - 1
        var continue = true
        var trimmed_unprocessed_path = unprocessed_path
        var slotX_indexes : List[Int] = List()
        
        // trim from left-hand side
        while (continue)
        {
            if (multi_word_group(unprocessed_path(i)._1) == multi_word_group(unprocessed_path(i)._2))
            {
                // delete left-most element
                trimmed_unprocessed_path = trimmed_unprocessed_path.drop(1)
                
                if (i==0)
                    slotX_indexes = multi_word_group(unprocessed_path(i)._1)
                
                if (i == last_idx)
                    continue = false
                else
                    i += 1
            }
            else
                continue = false
        }
        
        if (trimmed_unprocessed_path.length == 0)
            return (trimmed_unprocessed_path , -1 , -1 , -1 , -1)
        
        if (slotX_indexes.length == 0)
        {
            if (unprocessed_path(0)._4 == "<")
                slotX_indexes = unprocessed_path(0)._2 :: slotX_indexes
            else
                slotX_indexes = unprocessed_path(0)._1 :: slotX_indexes
        }
        
        val slotX_start_idx = slotX_indexes.min
        val slotX_end_idx   = slotX_indexes.max + 1
        
        //trim from right-hand side
        last_idx = trimmed_unprocessed_path.length - 1
        i = last_idx
        continue = true
        var u_p = trimmed_unprocessed_path
        var slotY_indexes : List[Int] = List()
        
        while (continue)
        {
            if (multi_word_group(u_p(i)._1) == multi_word_group(u_p(i)._2))
            {
                //delete right-most element
                trimmed_unprocessed_path = trimmed_unprocessed_path.dropRight(1)
                
                if (i==last_idx)
                    slotY_indexes = multi_word_group(u_p(i)._1)
                    
                if (i == 0)
                    continue = false
                else
                    i -= 1
            }
            else
                continue = false
        }
        
        if (slotY_indexes.length == 0)
        {
            if (u_p(last_idx)._4 == ">")
                slotY_indexes = u_p(last_idx)._2 :: slotY_indexes
            else
                slotY_indexes = u_p(last_idx)._1 :: slotY_indexes
        }
        
        val slotY_start_idx = slotY_indexes.min
        val slotY_end_idx   = slotY_indexes.max + 1
        
        return (trimmed_unprocessed_path , slotX_start_idx , slotX_end_idx , slotY_start_idx , slotY_end_idx)
    }
    
    
    def create_multi_word_groups(entities:Array[String] , namedEntityClasses:List[String]) : Map[Int , List[Int]] =
    {
        var multi_word_group : Map[Int , List[Int]] = Map()
        
        var current_list : List[Int] = List(0)
        
        for (i <- 1 until entities.length)
        {
            if ((namedEntityClasses.contains(entities(i))) && (entities(i) == entities(i-1)))
                current_list = i :: current_list
            
            else
            {
                for (j <- current_list)
                    multi_word_group(j) = current_list
                
                current_list = List(i)
            }
        }
        
        for (j <- current_list)
            multi_word_group(j) = current_list
        
        return multi_word_group
    }
    
    
    def is_valid_BERT_token_index(slotX_start_char_idx:Int , slotX_end_char_idx:Int ,
                                  slotY_start_char_idx:Int , slotY_end_char_idx:Int , 
                                  sentence_chars_to_BERTtokens_indexes:Map[Int , Map[Int,Int]] , 
                                  sen_id:Int , BERT_MAX_TOKEN_INDEX:Int) : Boolean =
    {
        var BERT_token_index = -1
        
        for (char_index <- List(slotX_start_char_idx , slotX_end_char_idx-1 , slotY_start_char_idx , slotY_end_char_idx-1))
        {
            BERT_token_index = sentence_chars_to_BERTtokens_indexes(sen_id)(char_index)
        
            if ((BERT_token_index<0) || (BERT_token_index >= BERT_MAX_TOKEN_INDEX))
                return false
        }
        
        return true
    }
    
    
    def get_char_span_in_sentence(start_word_index:Int , end_word_index:Int , words:Array[String]) : (Int , Int) =
    {
        var s = words.slice(0 , start_word_index+1).mkString(" ")
        
        var start_char_index = s.length - words(start_word_index).length
        
        s = words.slice(0 , end_word_index).mkString(" ")
        
        var end_char_index = s.length
        
        return (start_char_index , end_char_index)
    }
}
