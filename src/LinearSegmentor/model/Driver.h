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
		_modelparams.exportModelParams(_ada);
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
		_pcg->forward(&sentences[idx], _hyperparams.dicts, &goldACs[idx]);

		int seq_size = sentences[idx].size();
		_eval.overall_label_count += seq_size + 1;
		cost += loss(num);

		_pcg->backward();

		std::cout << seq_size << " " << _pcg->outputs.size() << std::endl;
    }

    return cost;
  }

  void decode(const std::vector<string>& sentence, vector<string>& words){
	  _pcg->forward(&sentence, _hyperparams.dicts);
	  predict(words);
  }

  void updateModel() {
	  _ada.update();
  }

  void writeModel();

  void loadModel();

private:
	// max-margin
	dtype loss(int num){
		int advancedStep = _pcg->outputs.size();
		_eval.correct_label_count += advancedStep;
		CStateItem* pBestGenerator = NULL;
		CStateItem* pGoldGenerator = NULL;
		CStateItem* pGenerator;
		
		for (int idx = 0; idx < _pcg->outputs[advancedStep - 1].size(); idx++){
			pGenerator = _pcg->outputs[advancedStep - 1][idx];
			if (pBestGenerator == NULL || pGenerator->_score.val(0, 0) > pBestGenerator->_score.val(0, 0)){
				pBestGenerator = pGenerator;
			}
			if (pGenerator->_bGold){
				if (pGoldGenerator != NULL){
					std::cout << "bug" << std::endl;
				}
				pGoldGenerator = pGenerator;
			}
		}

		if (pGoldGenerator != pBestGenerator){
			if (pGoldGenerator->_score.loss.size() == 0){
				pGoldGenerator->_score.loss = Mat::Zero(1, 1);
			}
			pGoldGenerator->_score.loss(0, 0) = -1.0 / num;
			pGoldGenerator->_score.lock--;

			if (pBestGenerator->_score.loss.size() == 0){
				pBestGenerator->_score.loss = Mat::Zero(1, 1);
			}
			pBestGenerator->_score.loss(0, 0) = 1.0 / num;
			pBestGenerator->_score.lock--;

			return 1.0;
		}

		return 0.0;
	}

	void predict(std::vector<std::string>& words){
		int advancedStep = _pcg->outputs.size();
		CStateItem* pBestGenerator = NULL;
		CStateItem* pGenerator = NULL;
		for (int idx = 0; idx < _pcg->outputs[advancedStep - 1].size(); idx++){
			pGenerator = _pcg->outputs[advancedStep - 1][idx];
			if (pBestGenerator == NULL || pGenerator->_score.val(0, 0) > pBestGenerator->_score.val(0, 0)){
				pBestGenerator = pGenerator;
			}
		}

		pBestGenerator->getSegResults(words);

	}


	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

};

#endif /* SRC_Driver_H_ */
