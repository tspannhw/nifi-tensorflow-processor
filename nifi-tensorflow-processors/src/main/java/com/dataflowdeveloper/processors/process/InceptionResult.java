package com.dataflowdeveloper.processors.process;

import java.io.Serializable;

/**
 * 
 * @author tspann
 *
 */
public class InceptionResult implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7558933631802853271L;
	
	private String label;
	private String probability;
	private int rank;
	
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("InceptionResult [label=");
		builder.append(label);
		builder.append(", probability=");
		builder.append(probability);
		builder.append(", rank=");
		builder.append(rank);
		builder.append("]");
		return builder.toString();
	}
	
	
	public InceptionResult() {
		super();
	}


	public InceptionResult(String label, String probability, int rank) {
		super();
		this.label = label;
		this.probability = probability;
		this.rank = rank;
	}


	public String getLabel() {
		return label;
	}
	public void setLabel(String label) {
		this.label = label;
	}
	public String getProbability() {
		return probability;
	}
	public void setProbability(String probability) {
		this.probability = probability;
	}
	public int getRank() {
		return rank;
	}
	public int getDisplayRank() {
		return rank + 1;
	}
	public void setRank(int rank) {
		this.rank = rank;
	}

}