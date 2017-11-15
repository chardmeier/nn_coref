package edu.harvard.nlp.moarcoref;

import java.io.File;
import java.util.Arrays;
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
			int[][] allfeats = new int[s * (s-1) / 2][];
			int maxlen = 0;
			for(int i = 0, k = 0; i < s; i++) {
				for(int j = 0; j < i; j++, k++) {
					Seq<Object> feats = dg.cachedFeats()[i][j];
					allfeats[k] = new int[feats.size()];
					if(feats.size() > maxlen)
						maxlen = feats.size();
					Iterator<Object> iter2 = feats.iterator();
					int l = 0;
					while(iter2.hasNext())
						allfeats[k][l++] = (int) iter2.next();
				}
			}

			for(int i = 0; i < allfeats.length; i++)
				if(allfeats[i].length < maxlen)
					allfeats[i] = Arrays.copyOf(allfeats[i], maxlen);
					
			h5.writeIntMatrix(Integer.toString(doc), allfeats);
			doc++;
		}
		h5.close();
	}
}
