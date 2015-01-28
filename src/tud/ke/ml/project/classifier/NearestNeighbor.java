package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import tud.ke.ml.project.framework.classifier.ANearestNeighbor;
import tud.ke.ml.project.util.Pair;
import weka.core.converters.ArffLoader;

/**
 * This implementation assumes the class attribute is always available (but probably not set)
 * @author cwirth
 *
 */
public class NearestNeighbor extends ANearestNeighbor {
	
	protected double[] scaling;
	protected double[] translation;
	
	private List<List<Object>> traindata;
	
	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> map;
		//System.out.println(subset);
		if (isInverseWeighting()){
			map = getWeightedVotes(subset);
		} else {
			map = getUnweightedVotes(subset);			
		}
		return getWinner(map);
	}
	@Override
	protected void learnModel(List<List<Object>> traindata) {
				this.traindata = traindata;
				//for (int i = 0; i < traindata.size(); i++){
				//	System.out.println("traindata : " + traindata.get(i));					
				//}
	}
	@Override
	protected Map<Object, Double> getUnweightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> unweightedVotes = new HashMap<Object, Double>();
		for (Pair<List<Object>, Double> pair : subset){
			Object classAttributeValue = pair.getA().get(pair.getA().size()-1);
			if (unweightedVotes.containsKey(classAttributeValue)){
				unweightedVotes.put(classAttributeValue, unweightedVotes.get(classAttributeValue)+1);
			} else {
				unweightedVotes.put(classAttributeValue, (double) 1);
			}
		}
		return unweightedVotes;
	}
	@Override
	protected Map<Object, Double> getWeightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> weightedVotes = new HashMap<Object, Double>();
		for (Pair<List<Object>, Double> pair : subset){
			Object classAttributeValue = pair.getA().get(pair.getA().size()-1);
			if (weightedVotes.containsKey(classAttributeValue)){
				weightedVotes.put(classAttributeValue, weightedVotes.get(classAttributeValue)+(1/(pair.getB()/**pair.getB()*/)));
			} else {
				weightedVotes.put(classAttributeValue, 1/(pair.getB()/**pair.getB()*/));
			}
		}
		return weightedVotes;
	}
	@Override
	protected Object getWinner(Map<Object, Double> votesFor) {
		List<Map.Entry<Object, Double>> maxEntries = new ArrayList<Map.Entry<Object, Double>>();
		Map.Entry<Object, Double> maxEntry = null;
		System.out.println();
		//for (Object o : votesFor.keySet()){
		//	System.out.println("class : " + o + " - weight : " + votesFor.get(o));
		//}
		for (Map.Entry<Object, Double> entry : votesFor.entrySet())
		{
			if (maxEntries.isEmpty() || entry.getValue().compareTo(maxEntries.get(0).getValue()) == 0)
		    {
				//System.out.println(entry.getValue());
			    maxEntries.add( entry);		    		
		    } else if (entry.getValue().compareTo(maxEntries.get(0).getValue()) > 0)
		    {
				maxEntries.clear();
				//System.out.println(entry.getValue());
			    maxEntries.add( entry);		    		
		    }
		}
		Random r = new Random();
    	int n = r.nextInt(maxEntries.size());
    	maxEntry = maxEntries.get(n);
    	//System.out.println("winner : " + maxEntry);
		return maxEntry.getKey();
	}
	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> testdata) {
		List<Pair<List<Object>, Double>> nearest = new ArrayList<Pair<List<Object>, Double>>() ;
		double distance;
		double[][] factors = normalizationScaling();
		scaling = new double[factors.length];
		translation = new double[factors.length];
		for (int i=0; i< factors.length; i++){
			scaling[i] = factors[i][0];
			translation[i] = factors[i][1];			
		}
		for (int i=0; i< traindata.size(); i++){
			if (getMetric()==0){
				distance = determineManhattanDistance(traindata.get(i), testdata);
			} else {
				distance = determineEuclideanDistance(traindata.get(i), testdata);				
			}
			nearest.add(new Pair(traindata.get(i), distance));
		}
		Collections.sort(nearest);
		//System.out.println();
		//for (int i = 0; i < nearest.size(); i++)
		//System.out.println(nearest.get(i).getB());
		int NbNearest = getkNearest();
		//if (NbNearest < nearest.size()) System.out.println("taille");
		//if (nearest.get(NbNearest-1).getB().equals(nearest.get(NbNearest).getB())) System.out.println("egal !!!");
		while ((NbNearest < nearest.size()) && (nearest.get(NbNearest-1).getB().equals(nearest.get(NbNearest).getB()))){
			NbNearest++;
		}
		//System.out.println("nbNearest : " + Math.min(NbNearest,nearest.size()));
		//System.out.println("distance1 : " + nearest.get(NbNearest-1).getB() + "  distance2 : " + nearest.get(NbNearest).getB());
		return nearest.subList(0, Math.min(NbNearest,nearest.size()));
	}
	@Override
	protected double determineManhattanDistance(List<Object> instance1,
			List<Object> instance2) {
		double distance = 0;
		//System.out.println(instance1);
		//System.out.println(instance2);
		
		for (int i=0; i<instance1.size()-1; i++){
			if (instance1.get(i) instanceof Double ){
				if (scaling[i]!=0){
					distance += Math.abs((((Double)instance1.get(i)-translation[i])/scaling[i])-(((Double)instance2.get(i)-translation[i])/scaling[i])) ;
				}
			} else {
				if (!instance1.get(i).equals(instance2.get(i))){
					distance += 1;
				}
			}
		}
		//System.out.println("manhattan : " + distance);
		
		return distance;
	}
	@Override
	protected double determineEuclideanDistance(List<Object> instance1,
			List<Object> instance2) {
		double distance = 0;
		for (int i=0; i<instance1.size()-1; i++){
			if (instance1.get(i) instanceof Double ){
				if (scaling[i]!=0){
					Double a =((((Double)instance1.get(i)-translation[i])/scaling[i]));
					Double b = (((Double)instance2.get(i)-translation[i])/scaling[i]);
					//if (a > 1 || a < 0 || b > 1 || b < 0){
					//	System.out.println("aNN : " + (Double)instance1.get(i) + "   bNN : " + (Double)instance2.get(i));
					//	System.out.println("aN : " + a + "   bN : " + b);
					//	System.out.println("translation : " + translation[i] + "   scaling : " + scaling[i]);						
					//}
					
					distance += ((((Double)instance1.get(i)-translation[i])/scaling[i])-(((Double)instance2.get(i)-translation[i])/scaling[i]))*((((Double)instance1.get(i)-translation[i])/scaling[i])-(((Double)instance2.get(i)-translation[i])/scaling[i]));				
				}
			} else {
				if (!instance1.get(i).equals(instance2.get(i))){
					distance += 1;
				}
			}
		}
		return Math.sqrt(distance);
	}
	@Override
	protected double[][] normalizationScaling() {
		int nbAttributes = traindata.get(0).size() - 1;
		double[][] factors = new double[nbAttributes][2];
		if (isNormalizing()){
			for (int attribut = 0; attribut < nbAttributes; attribut++){
				double max = Integer.MIN_VALUE;
				double min = Integer.MAX_VALUE;
				for (int i = 0; i < traindata.size(); i++){
					//System.out.print(traindata.get(i).get(attribut) + "   ");
					if (traindata.get(i).get(attribut) instanceof Double){
						double value = (Double) traindata.get(i).get(attribut);
						if ( value > max){
							max = value;
						}
						if (value < min){
							min = value;
						}
					}
				}
				//System.out.println();
				//System.out.println("max : " + max);
				//System.out.println("min : " + min);
				factors[attribut][0] = max - min; 
				factors[attribut][1] = min;					
			}			
		} else {
			for (int attribut = 0; attribut < nbAttributes; attribut++){
				for (int i = 0; i < traindata.size(); i++){
					factors[attribut][0] = 1; 
					factors[attribut][1] = 0;					
				}
			}
		}
		return factors;
	}
	@Override
	protected String[] getMatrikelNumbers() {
		return new String[] {"2709749","2878405"};
	}

}
