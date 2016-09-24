/*
 * Driver.h
 *
 *  Created on: Jan 25, 2016
 *      Author: mszhang
 */

#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include "N3L.h"
#include "State.h"
#include "ActionedNodes.h"
#include "Action.h"
#include "ComputionGraph.h"


using namespace nr;
using namespace std;

//re-implementation of Yue and Clark ACL (2007)
class Driver {
public:
  Driver() {
	  _pcg = NULL;
  }

  ~Driver() {
	  if (_pcg != NULL)
		  delete _pcg;
	  _pcg = NULL;
  }

public:
	ComputionGraph*  _pcg;
	ModelParams _modelparams;  // model parameters
	HyperParams _hyperparams;

	Metric _eval;
	CheckGrad _checkgrad;
	ModelUpdate _ada;  // model update

public:

	inline void initial() {
		if (!_hyperparams.bValid()){
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}
		if (!_modelparams.initial(_hyperparams)){
			std::cout << "model parameter initialization Error, Please check!" << std::endl;
			return;
		}		
		_hyperparams.print();

		_pcg = new ComputionGraph();
		_pcg->initial(_modelparams, _hyperparams);

		setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
	}


public:
  dtype train(const std::vector<std::vector<string> >& sentences, const vector<vector<CAction> >& goldACs) {
    _eval.reset();
    dtype cost = 0.0;
	int num = sentences.size();
	for (int idx = 0; idx < num; idx++) {
		_pcg->forward(&sentences[idx], &goldACs[idx]);

		int seq_size = sentences[idx].size();
		_eval.overall_label_count += seq_size + 1;
		cost += loss(num);  

		_pcg->backward();

    }

    return cost;
  }

  void decode(const std::vector<string>& sentence, vector<string>& result){
	  _pcg->forward(&sentence);
	  predict(result);
  }

  void extractFeat(const std::vector<std::vector<string> >& sentences, const vector<vector<CAction> >& goldACs){
	  int num = sentences.size();
	  for (int idx = 0; idx < num; idx++) {
		  _pcg->extractFeat(&sentences[idx], &goldACs[idx]);
	  }
  }

  void updateModel() {
	  if (_ada._params.empty()){
		  _modelparams.exportModelParams(_ada);
	  }
	  _ada.update();
  }

  void writeModel();

  void loadModel();

private:
	// max-margin
	dtype loss(int num){
		int step = _pcg->outputs.size();
		_eval.correct_label_count += step;
		static PNode pBestNode = NULL;
		static PNode pGoldNode = NULL;
		static PNode pCurNode;
		
		int offset = _pcg->outputs[step - 1].size();
		for (int idx = 0; idx < offset; idx++){
			pCurNode = _pcg->outputs[step - 1][idx].in;
			if (pBestNode == NULL || pCurNode->val.coeffRef(0) > pBestNode->val.coeffRef(0)){
				pBestNode = pCurNode;
			}
			if (_pcg->outputs[step - 1][idx].bGold){
				pGoldNode = pCurNode;
			}
		}

		if (pGoldNode != pBestNode){
			if (pGoldNode->loss.size() == 0){
				pGoldNode->loss = Mat::Zero(1, 1);
			}
			pGoldNode->loss.coeffRef(0) = -1.0 / num;

			if (pBestNode->loss.size() == 0){
				pBestNode->loss = Mat::Zero(1, 1);
			}
			pBestNode->loss.coeffRef(0) = 1.0 / num;

			return 1.0;
		}

		return 0.0;
	}
	
	void predict(vector<string>& result){
		int step = _pcg->outputs.size();
		_pcg->states[step - 1][0].getSegResults(result);
	}


	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

};

#endif /* SRC_Driver_H_ */
