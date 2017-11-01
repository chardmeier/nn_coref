#!/bin/sh
exec scala -J-Xmx16G -classpath "moarcoref-assembly-1.jar:lib/futile.jar:lib/BerkeleyParser-1.7.jar" "$0" "$@"
!#

import java.io.File
import java.io.PrintWriter
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import edu.berkeley.nlp.coref.NumberGenderComputer
import edu.berkeley.nlp.coref._
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.util.Logger

object GetMentionSpans {

  // same as original, except we sort files by names so we can dump features and then repredict  
  def loadRawConllDocs(path: String, size: Int, gold: Boolean): Seq[ConllDoc] = {
    val suffix = if (gold) "gold_conll" else Driver.docSuffix;
    Logger.logss("Loading " + size + " docs from " + path + " ending with " + suffix);
    val files = new File(path).listFiles().filter(file => file.getAbsolutePath.endsWith(suffix)); //.sorted;
    val reader = new ConllDocReader(Driver.lang);
    val docs = new ArrayBuffer[ConllDoc];
    var docCounter = 0;
    var fileIdx = 0;
    while (fileIdx < files.size && (size == -1 || docCounter < size)) {
      val newDocs = reader.readConllDocs(files(fileIdx).getAbsolutePath);
      docs ++= newDocs;
      docCounter += newDocs.size
      fileIdx += 1;
    }
    val numDocs = if (size == -1) docs.size else Math.min(size, files.size);
    Logger.logss(docs.size + " docs loaded from " + fileIdx + " files, retaining " + numDocs);
    if (docs.size == 0) {
      Logger.logss("WARNING: Zero docs loaded...double check your paths unless you meant for this happen");
    }
    val docsToUse = docs.slice(0, numDocs);
    
    docsToUse;
  }
  
  // same as in original
  def loadCorefDocs(path: String, size: Int, numberGenderComputer: NumberGenderComputer, gold: Boolean): Seq[CorefDoc] = {
    val docs = loadRawConllDocs(path, size, gold);
    val assembler = CorefDocAssembler(Driver.lang, Driver.useGoldMentions);
    val mentionPropertyComputer = new MentionPropertyComputer(numberGenderComputer);
    val corefDocs = docs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    CorefDoc.checkGoldMentionRecall(corefDocs);
    corefDocs;
  }
 
  def main(args: Array[String]) {
    val devPath = args(0);    
    val ngPath = args(1);
    val outFile = args(2);

    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(ngPath);
    println("loading coref docs");
    val devDGs = loadCorefDocs(devPath, -1, numberGenderComputer, false).map(new DocumentGraph(_, false)).sortBy(_.corefDoc.rawDoc.printableDocName);    
    println("done");
    val out = new PrintWriter(new File(outFile));
    for (i <- 0 until devDGs.size) {
      val mentions = devDGs(i).getMentions();
      for (j <- 0 until mentions.size) {
        val ment = mentions(j);
        out.println(s"$i\t$j\t${ment.sentIdx}\t${ment.startIdx}\t${ment.endIdx}");
      }
    } 
    out.close();
  }
 
}

GetMentionSpans.main(args)
