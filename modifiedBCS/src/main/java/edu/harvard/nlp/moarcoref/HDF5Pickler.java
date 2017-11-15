package edu.harvard.nlp.moarcoref;

import java.io.File;
import scala.collection.Iterator;
import scala.collection.Seq;
import ch.systemsx.cisd.hdf5.HDF5FactoryProvider;
import ch.systemsx.cisd.hdf5.IHDF5SimpleWriter;
import edu.berkeley.nlp.coref.DocumentGraph;

public class HDF5Pickler {
	public static void writePWFeats(Seq<DocumentGraph> docGraphs, int biasFeature, String outname) {
		IHDF5SimpleWriter h5 = HDF5FactoryProvider.get().open(new File(outname));
		int doc = 0;
		Iterator<DocumentGraph> iter = docGraphs.iterator();
		while(iter.hasNext()) {
			DocumentGraph dg = iter.next();
			int s = dg.size();
			int[] allfeats = new int[s * (s-1) / 2];
			for(int i = 0; i < s; i++) {
				int k = 0;
				for(int j = 0; j < i; j++) {
					Seq<Object> feats = dg.cachedFeats()[i][j];
					Iterator<Object> iter2 = feats.iterator();
					while(iter2.hasNext()) {
						Object o = iter2.next();
						//System.err.println(o.getClass().getName());
						allfeats[k++] = (int) o;
					}
				}
			}
			h5.writeIntArray(Integer.toString(doc), allfeats);
			doc++;
		}
		h5.close();
	}
}
