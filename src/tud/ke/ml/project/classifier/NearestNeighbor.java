package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.Collections;
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
	}
	@Override
	protected Map<Object, Double> getUnweightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		
		return null;
	}
	@Override
	protected Map<Object, Double> getWeightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		
		return null;
	}
	@Override
	protected Object getWinner(Map<Object, Double> votesFor) {
		Map.Entry<Object, Double> maxEntry = null;
		for (Map.Entry<Object, Double> entry : votesFor.entrySet())
		{
			if (maxEntry == null || entry.getValue().compareTo(maxEntry.getValue()) > 0)
		    {
		        maxEntry = entry;
		    }
		}
		return maxEntry.getKey();
	}
	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> testdata) {
		List<Pair<List<Object>, Double>> nearest = null;
		double distance;
		for (int i=0; i< traindata.size(); i++){
			if (getMetric()==0){
				distance = determineManhattanDistance(traindata.get(i), testdata);
			} else {
				distance = determineEuclideanDistance(traindata.get(i), testdata);				
			}
			nearest.set(i, new Pair(traindata.get(i), distance));
		}
		Collections.sort(nearest);
		return nearest.subList(0, getkNearest()-1);
	}
	@Override
	protected double determineManhattanDistance(List<Object> instance1,
			List<Object> instance2) {
		double distance = 0;
		for (int i=0; i<instance1.size(); i++){
			if (instance1.get(i) instanceof Double ){
				distance += Math.abs((Double)instance1.get(i)-(Double)instance2.get(i));				
			} else {
				if (!instance1.get(i).equals(instance2.get(i))){
					distance += 1;
				}
			}
		}
		return distance;
	}
	@Override
	protected double determineEuclideanDistance(List<Object> instance1,
			List<Object> instance2) {
		double distance = 0;
		for (int i=0; i<instance1.size(); i++){
			if (instance1.get(i) instanceof Double ){
				distance += ((Double)instance1.get(i)-(Double)instance2.get(i))*((Double)instance1.get(i)-(Double)instance2.get(i));				
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
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	protected String[] getMatrikelNumbers() {
		return new String[] {"2709749","2878405"};
	}

}
