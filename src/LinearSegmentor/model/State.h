/*
 * State.h
 *
 *  Created on: Oct 1, 2015
 *      Author: mszhang
 */

#ifndef SEG_STATE_H_
#define SEG_STATE_H_

#include "ModelParams.h"
#include "Action.h"
#include "ActionedNodes.h"
#include "AtomFeatures.h"
#include "Utf.h"

class CStateItem {
public:
	std::string _strlastWord;
	int _lastWordStart;
	int _lastWordEnd;
	CStateItem *_prevStackState;
	CStateItem *_prevState;
	int _nextPosition;

	const std::vector<std::string> *_pCharacters;
	int _characterSize;

	CAction _lastAction;
	PAddNode _score;
	int _wordnum;

	// features
	ActionedNodes _current;  // features current used
	AtomFeatures _atomFeat;  //features will be used for future

public:
	bool _bStart; // whether it is a start state
	bool _bGold; // for train

public:
	CStateItem() {
		clear();
	}


	virtual ~CStateItem(){
		clear();
	}

	void initial(ModelParams& params, HyperParams& hyparams) {
		_current.initial(params, hyparams);
	}

	void setInput(const std::vector<std::string>* pCharacters) {
		_pCharacters = pCharacters;
		_characterSize = pCharacters->size();
	}

	void clear() {
		_strlastWord = "";
		_lastWordStart = -1;
		_lastWordEnd = -1;
		_prevStackState = 0;
		_prevState = 0;
		_nextPosition = 0;
		_pCharacters = 0;
		_characterSize = 0;
		_lastAction.clear();
		_wordnum = 0;
		_bStart = true;
		_bGold = true;
	}


	const CStateItem* getPrevStackState() const{
		return _prevStackState;
	}

	const CStateItem* getPrevState() const{
		return _prevState;
	}

	std::string getLastWord() {
		return _strlastWord;
	}


public:
	//only assign context
	void separate(CStateItem* next){
		if (_nextPosition >= _characterSize) {
			std::cout << "separate error" << std::endl;
			return;
		}
		next->_strlastWord = (*_pCharacters)[_nextPosition];
		next->_lastWordStart = _nextPosition;
		next->_lastWordEnd = _nextPosition;
		next->_prevStackState = this;
		next->_prevState = this;
		next->_nextPosition = _nextPosition + 1;
		next->_pCharacters = _pCharacters;
		next->_characterSize = _characterSize;
		next->_wordnum = _wordnum + 1;
		next->_lastAction.set(CAction::SEP);
	}

	//only assign context
	void finish(CStateItem* next){
		if (_nextPosition != _characterSize) {
			std::cout << "finish error" << std::endl;
			return;
		}
		next->_strlastWord = _strlastWord;
		next->_lastWordStart = _lastWordStart;
		next->_lastWordEnd = _lastWordEnd;
		next->_prevStackState = _prevStackState;
		next->_prevState = this;
		next->_nextPosition = _nextPosition + 1;
		next->_pCharacters = _pCharacters;
		next->_characterSize = _characterSize;
		next->_wordnum = _wordnum + 1;
		next->_lastAction.set(CAction::FIN);
	}

	//only assign context
	void append(CStateItem* next){
		if (_nextPosition >= _characterSize) {
			std::cout << "append error" << std::endl;
			return;
		}
		next->_strlastWord = _strlastWord + (*_pCharacters)[_nextPosition];
		next->_lastWordStart = _lastWordStart;
		next->_lastWordEnd = _nextPosition;
		next->_prevStackState = _prevStackState;
		next->_prevState = this;
		next->_nextPosition = _nextPosition + 1;
		next->_pCharacters = _pCharacters;
		next->_characterSize = _characterSize;
		next->_wordnum = _wordnum;
		next->_lastAction.set(CAction::APP);
	}

	void move(CStateItem* next, const CAction& ac){
		if (ac.isAppend()) {
			append(next);
		}
		else if (ac.isSeparate()) {
			separate(next);
		}
		else if (ac.isFinish()) {
			finish(next);
		}
		else {
			std::cout << "error action" << std::endl;
		}

		next->_bStart = false;
		next->_bGold = false;
	}

	bool IsTerminated() const {
		if (_lastAction.isFinish())
			return true;
		return false;
	}

	//partial results
	void getSegResults(std::vector<std::string>& words) const {
		words.clear();
		words.insert(words.begin(), _strlastWord);
		const CStateItem *prevStackState = _prevStackState;
		while (prevStackState != 0 && prevStackState->_wordnum > 0) {
			words.insert(words.begin(), prevStackState->_strlastWord);
			prevStackState = prevStackState->_prevStackState;
		}
	}


	void getGoldAction(const std::vector<std::string>& segments, CAction& ac) const {
		if (_nextPosition == _characterSize) {
			ac.set(CAction::FIN);
			return;
		}
		if (_nextPosition == 0) {
			ac.set(CAction::SEP);
			return;
		}

		if (_nextPosition > 0 && _nextPosition < _characterSize) {
			// should have a check here to see whether the words are match, but I did not do it here
			if (_strlastWord.length() == segments[_wordnum - 1].length()) {
				ac.set(CAction::SEP);
				return;
			}
			else {
				ac.set(CAction::APP);
				return;
			}
		}

		ac.set(CAction::NO_ACTION);
		return;
	}

	// we did not judge whether history actions are match with current state.
	void getGoldAction(const CStateItem* goldState, CAction& ac) const{
		if (_nextPosition > goldState->_nextPosition || _nextPosition < 0) {
			ac.set(CAction::NO_ACTION);
			return;
		}
		const CStateItem *prevState = goldState->_prevState;
		CAction curAction = goldState->_lastAction;
		while (_nextPosition < prevState->_nextPosition) {
			curAction = prevState->_lastAction;
			prevState = prevState->_prevState;
		}
		return ac.set(curAction._code);
	}

	void getCandidateActions(vector<CAction> & actions) const{
		actions.clear();
		static CAction ac;
		if (_nextPosition == 0){
			ac.set(CAction::SEP);
			actions.push_back(ac);
		}
		else if (_nextPosition == _characterSize){
			ac.set(CAction::FIN);
			actions.push_back(ac);
		}
		else if (_nextPosition > 0 && _nextPosition < _characterSize){
			ac.set(CAction::SEP);
			actions.push_back(ac);
			ac.set(CAction::APP);
			actions.push_back(ac);
		}
		else{

		}

	}

	inline std::string str() const{
		stringstream curoutstr;
		curoutstr << "score: " << _score.val(0, 0) << " ";
		curoutstr << "seg:";
		std::vector<std::string> words;
		getSegResults(words);
		for (int idx = 0; idx < words.size(); idx++){
			curoutstr << " " << words[idx];
		}

		return curoutstr.str();
	}


public:
	inline void computeScore(Graph *cg){
		if (_bStart){
			_current.forward(cg, _lastAction, _prevState->_atomFeat, NULL);
		}
		else{
			_current.forward(cg, _lastAction, _prevState->_atomFeat, &(_prevState->_score));
		}
	}

	inline void prepare(const hash_set<string>& dicts){
		_atomFeat.str_C0 = _nextPosition < _characterSize ? _pCharacters->at(_nextPosition) : nullkey;
		_atomFeat.str_1C = _nextPosition > 0 ? _pCharacters->at(_nextPosition - 1) : nullkey;
		_atomFeat.str_2C = _nextPosition > 1 ? _pCharacters->at(_nextPosition - 2) : nullkey;

		_atomFeat.str_CT0 = _nextPosition < _characterSize ? wordtype(_atomFeat.str_C0) : nullkey;
		_atomFeat.str_1CT = _nextPosition > 0 ? wordtype(_atomFeat.str_1C) : nullkey;
		_atomFeat.str_2CT = _nextPosition > 1 ? wordtype(_atomFeat.str_2C) : nullkey;

		_atomFeat.str_1W = _lastWordEnd == -1 ? nullkey : _strlastWord;
		_atomFeat.str_1Wc0 = _lastWordEnd == -1 ? nullkey : _pCharacters->at(_lastWordStart);
		_atomFeat.str_1WD = _lastWordEnd == -1 ? nullkey : (dicts.find(_atomFeat.str_1W) != dicts.end() ? "iv" : "oov");
		{
			int length = _lastWordEnd - _lastWordStart + 1;
			if (length > 5)
				length = 5;
			stringstream curss;
			curss << length;
			_atomFeat.str_1WL = _lastWordEnd == -1 ? nullkey : curss.str();
			_atomFeat.int_1WL = length;
		}
		if (_lastWordEnd == -1){
			_atomFeat.str_1Wci.clear();
		}
		else{
			_atomFeat.str_1Wci.clear();
			for (int idx = _lastWordStart; idx < _lastWordEnd; idx++){
				_atomFeat.str_1Wci.push_back(_pCharacters->at(idx));
			}
		}


		int last2WordStart = _prevStackState == 0 ? -1 : _prevStackState->_lastWordStart;
		int last2WordEnd = _prevStackState == 0 ? -1 : _prevStackState->_lastWordEnd;
		_atomFeat.str_2W = last2WordEnd == -1 ? nullkey : _prevStackState->_strlastWord;
		_atomFeat.str_2Wc0 = last2WordEnd == -1 ? nullkey : _pCharacters->at(last2WordStart);
		_atomFeat.str_2Wcn = last2WordEnd == -1 ? nullkey : _pCharacters->at(last2WordEnd);
		{
			int length = last2WordEnd - last2WordStart + 1;
			if (length > 5)
				length = 5;
			stringstream curss;
			curss << length;
			_atomFeat.str_2WL = last2WordEnd == -1 ? nullkey : curss.str();
		}
	}
};


class CScoredState {
public:
	CStateItem *item;
	dtype score;

public:
	CScoredState() : item(0), score(0) {
	}

public:
	bool operator <(const CScoredState &a1) const {
		return score < a1.score;
	}
	bool operator >(const CScoredState &a1) const {
		return score > a1.score;
	}
	bool operator <=(const CScoredState &a1) const {
		return score <= a1.score;
	}
	bool operator >=(const CScoredState &a1) const {
		return score >= a1.score;
	}
};

class CScoredState_Compare {
public:
	int operator()(const CScoredState &o1, const CScoredState &o2) const {

		if (o1.score < o2.score)
			return -1;
		else if (o1.score > o2.score)
			return 1;
		else
			return 0;
	}
};


#endif /* SEG_STATE_H_ */
