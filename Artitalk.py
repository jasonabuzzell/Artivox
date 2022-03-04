import re
import os
import json
import time
import random
import threading
import tkinter as tk
from colorama import Fore

# Third-party packages
import nltk
import numpy as np
import text2emotion as te
from pattern.text import en
import pandas as pd
import statsmodels.api as sm
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)

# Global variables
timer = 0
first_row = True
training = True
reset_flag = False

CY = Fore.CYAN
LW = Fore.LIGHTWHITE_EX
GR = Fore.GREEN

converse = {
    'Previous Speaker': '',
    'Previous Speak': '',
    'Time': {},
    'Multiple': {},
    'Silent': {},
    'Hold': {}
}

quant = {
    'Meaning': ['CC', 'CD', 'DT', 'EX', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNS', 'PDT', 'POS', 'PRP',
                'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
                'WRB', '$', '', ')', '(', ',', '--', '.', ':', 'FW', 'NNPS', 'SYM', 'WP$', '``', "''"],
    'Tense': ['Future_Perfect_Continuous', 'Future_Continuous', 'Future_Perfect', 'Past_Perfect_Continuous',
              'Present_Perfect_Continuous', 'Future_Indefinite', 'Past_Continuous', 'Past_Perfect',
              'Present_Continuous', 'Present_Continuous2', 'Present_Perfect', 'Present_Perfect2', 'Past_Indefinite',
              'Present_Indefinite', 'Present_Indefinite2'],
    'Mood': ['Indicative', 'Imperative', 'Interrogative', 'Conditional', 'Subjunctive'],
}


# ---------------------------------------------------------------------------------------------------------------------
# CLASSES

class Word:
    def __init__(self, word):
        self.word = word

    def insert(self, meaning):
        # Instead of a general dictionary, there should a word count for each character (only when scores of 1?)
        # If you need a general dictionary, find something in nltk or pattern
        # Also use this for user input when inhabiting a character
        if not os.path.exists('dictionary.json'):
            with open(f'dictionary.json', 'w') as f:
                json.dump({}, f)
        with open(f'dictionary.json', 'r') as f:
            dictionary = json.load(f)
        if self.word in dictionary:
            dictionary[self.word].append(meaning)
            dictionary[self.word] = list(set(dictionary[self.word]))
        else:
            dictionary[self.word] = [meaning]
        with open(f'dictionary.json', 'w') as f:
            json.dump(dictionary, f)


class Sentence:
    def __init__(self, sentence):
        self.sentence = sentence

    @staticmethod
    def tenses(meanings, mood):
        tenses = {
            'Future_Perfect_Continuous': ['MD', 'VB', 'VBN', 'VBG'],
            'Future_Continuous': ['MD', 'VB', 'VBG'],
            'Future_Perfect': ['MD', 'VB', 'VBN'],
            'Past_Perfect_Continuous': ['VBD', 'VBN', 'VBG'],
            'Present_Perfect_Continuous': ['VBP', 'VBZ', 'VBN', 'VBG'],
            'Future_Indefinite': ['MD', 'VB'],
            'Past_Continuous': ['VBD', 'VBG'],
            'Past_Perfect': ['VBD', 'VBN'],
            'Present_Continuous': ['VBZ', 'VBG'],
            'Present_Continuous2': ['VBP', 'VBG'],
            'Present_Perfect': ['VBZ', 'VBN'],
            'Present_Perfect2': ['VBP', 'VBN'],
            'Past_Indefinite': ['VBD'],
            'Present_Indefinite': ['VBZ'],
            'Present_Indefinite2': ['VBP']
        }

        i = 0
        text_pos = {}
        verbs = ['M', 'V']
        stoppers = ['PRP', 'DT', 'UH', 'TO']
        prev_tag = None
        for tag in meanings:
            if tag[1][0] in verbs:
                if prev_tag and prev_tag[1][0] in verbs:
                    text_pos[str(i)].append(tag[1])
                else:
                    i += 1
                    text_pos[str(i)] = [tag[1]]
                prev_tag = tag
            elif mood == 'Interrogative' and tag[1] in stoppers:
                pass
            else:
                prev_tag = tag

        copy = text_pos
        for verb_segment in list(copy):
            for tense in tenses:
                if text_pos[verb_segment] == tenses[tense]:
                    tense = tense.replace('2', '')
                    text_pos[verb_segment] = tense
                    break
            if type(text_pos[verb_segment]) == list:
                text_pos.pop(verb_segment)

        return text_pos

    @staticmethod
    def subjects(meanings):
        subjects = []
        for meaning in meanings:
            if meaning[1][0] == 'N' or 'PRP' in meaning[1]:
                subjects.append(meaning[0])

        return subjects

    def mood(self, words):
        questions = ['who', 'what', 'when', 'where', 'why', 'how']
        if not words:
            return 'Indicative'
        elif words[0].lower() in questions or '?' in words:
            return 'Interrogative'
        else:
            mood = en.mood(en.Sentence(en.parse(self.sentence)))
            return mood.capitalize()

    def find(self):
        sentence_dict = {}
        words = nltk.word_tokenize(self.sentence)
        meanings = nltk.pos_tag(words)
        for meaning in meanings:
            Word(meaning[0]).insert(meaning[1])
        sentence_dict['Meaning'] = meanings
        sentence_dict['Subject'] = self.subjects(meanings)
        sentence_dict['Mood'] = self.mood(words)
        sentence_dict['Tense'] = self.tenses(meanings, sentence_dict['Mood'])
        sentence_dict['Sentiment'] = SentimentIntensityAnalyzer().polarity_scores(self.sentence)
        sentence_dict['Emotion'] = te.get_emotion(self.sentence)
        sentence_dict['Word Count'] = len(meanings)

        return sentence_dict


class Text:
    def __init__(self, text):
        self.text = text

    def words(self):
        words = []
        sentences = nltk.sent_tokenize(self.text)
        for sentence in sentences:
            words.append(nltk.word_tokenize(sentence))

        return words

    def sentences(self):
        return nltk.sent_tokenize(self.text)

    def find(self):
        text_dict = {}
        sentences = nltk.sent_tokenize(self.text)
        if sentences:
            for sentence in sentences:
                text_dict[sentence] = Sentence(sentence).find()
        else:
            text_dict[self.text] = Sentence(self.text).find()

        return text_dict


class Character:
    def __init__(self, character):
        self.character = character

    @staticmethod
    def select():
        global LW

        if not os.path.exists('characters'):
            os.makedirs('characters')
        if not os.path.exists('characters/characters.json'):
            with open(f'characters/characters.json', 'w') as f:
                json.dump({}, f)

        with open(f'characters/characters.json', 'r') as f:
            characters = json.load(f)
        while True:
            selected = {}
            print(LW + 'Who is participating in this conversation? (e.g. [1], [2], [ira]')
            i = 1
            if not characters:
                characters = {}
            else:
                for character in characters:
                    print(LW + f'{i}. {character}')
                    i += 1
            print(LW + f'{i}. Create new character...')
            for char in input().split(', '):
                selected[char] = 'Non-user'
            if 'Create new' in selected:
                print(LW + f'What is the name of the new character?')
                characters.update({input(): {'Converse': {
                    'Time Mean': 8,
                    'Time Deviation': 5,
                    'Multiple': 3,
                    'Silent Mean': 4,
                    'Silent Deviation': 2,
                    'Separation Mean': 4,
                    'Separation Deviation': 2}}})
            elif len(selected) < 2:
                print(LW + f'Please enter at least two characters.\n')
            elif not all(x in characters for x in list(selected)):
                print(LW + f'Please make sure all characters are spelled correctly, case-sensitive.\n')
            else:
                print(LW + 'What character will you be using?')
                user = input()
                if not user:
                    break
                elif user not in characters:
                    print(LW + 'User not found among characters.\n')
                else:
                    selected[user] = 'User'
                    break

        for char in selected:
            if not os.path.exists(f'characters/{char}.json'):
                with open(f'characters/{char}.json', 'w') as f:
                    pref = {
                        'Greetings': {},
                        'Responses': {},
                        'Repeats': {},
                        'General': {'Count': 0, 'Word Count': 10, 'Sentence Count': 2, 'Verbosity': 5,
                                    'Sentiment|neg': 0.5, 'Sentiment|neu': 0.5, 'Sentiment|pos': 0.5,
                                    'Sentiment|compound': 0.5, 'Emotion|Happy': 0.5, 'Emotion|Angry': 0.5,
                                    'Emotion|Surprise': 0.5, 'Emotion|Sad': 0.5, 'Emotion|Fear': 0.5},
                        'Dictionary': {}
                    }

                    json.dump(pref, f)
        with open(f'characters/characters.json', 'w') as f:
            json.dump(characters, f)

        return selected

    @staticmethod
    def model_maker(text_dict, text_responses=None):
        model = {}
        MLR_model = {'Constant': []}

        # Reorganizes and quantizes all parameters except Subjects (which is handled separately)
        count = 1
        for response in text_dict:
            MLR_model['Constant'].append(0)
            i = 0
            for sentence in text_dict[response]:
                prev_model_tag = None
                for tag in text_dict[response][sentence]['Meaning']:
                    model_tag = 'Meaning-' + str(quant['Meaning'].index(tag[1]))
                    while prev_model_tag in model:
                        if '.' in prev_model_tag:
                            index = int(prev_model_tag[-1]) + 1
                            prev_model_tag = prev_model_tag[:-1] + str(index)
                        else:
                            prev_model_tag += '.2'
                    if prev_model_tag:
                        model[prev_model_tag] = quant['Meaning'].index(tag[1])
                    prev_model_tag = model_tag
                model['Mood-' + str(i)] = quant['Mood'].index(text_dict[response][sentence]['Mood'])
                model['Word Count-' + str(i)] = text_dict[response][sentence]['Word Count']
                for tense in text_dict[response][sentence]['Tense']:
                    model['Tense-' + str(i)] = quant['Tense'].index(text_dict[response][sentence]['Tense'][tense])
                for sentiment in text_dict[response][sentence]['Sentiment']:
                    sentiment_tag = 'Sentiment|' + sentiment + '-' + str(i)
                    model[sentiment_tag] = text_dict[response][sentence]['Sentiment'][sentiment]
                for emotion in text_dict[response][sentence]['Emotion']:
                    emotion_tag = 'Emotion|' + emotion + '-' + str(i)
                    model[emotion_tag] = text_dict[response][sentence]['Emotion'][emotion]
                i += 1
                model['Sentence Count'] = i

            if text_responses:
                model['Score'] = text_responses[response][0]
                model['Trials'] = text_responses[response][1]
            for parameter in model:
                if parameter in MLR_model:
                    MLR_model[parameter].append(model[parameter])
                else:
                    MLR_model[parameter] = []
                    i = 1
                    while i < count:
                        MLR_model[parameter].append('nan')
                        i += 1
                    MLR_model[parameter].append(model[parameter])
            # One more time to catch all parameters
            for parameter in MLR_model:
                if len(MLR_model[parameter]) < count:
                    MLR_model[parameter].append('nan')
            count += 1
            model = {}

        # Dealing with 'nan' values so MLR works
        for parameter in MLR_model:
            if 'nan' in MLR_model[parameter]:
                elements = {}
                for value in MLR_model[parameter]:
                    if value != 'nan':
                        elements[value] = MLR_model[parameter].count(value)
                keys = list(elements.keys())
                total = sum(list(elements.values()))
                weights = []
                for key in elements:
                    weights.append(elements[key] / total)
                for i in range(len(MLR_model[parameter])):
                    if MLR_model[parameter][i] == 'nan':
                        MLR_model[parameter][i] = np.random.choice(keys, p=weights)
            MLR_model[parameter] = np.asarray(MLR_model[parameter])

        return MLR_model

    def reconstruction(self, reconstruct, pref_words):
        global converse
        # Should be using converse far more often
        
        output = ''
        endings = ['.', '!', '?']

        with open(f'characters/{self.character}.json', 'r') as f:
            pref = json.load(f)
        with open(f'dictionary.json', 'r') as f:
            dictionary = json.load(f)

        for i in range(round(reconstruct['Sentence Count'])):
            sentence_dict = {'Sentence': '', 'Previous Word': '', 'Subjects': ''}
            for j in range(round(reconstruct['Word Count'])):
                max_score = ['', 0]
                for word in dictionary:
                    score = 0
                    meaning = dictionary[word]

                    if sentence_dict['Previous Word']:
                        prev_meaning = dictionary[sentence_dict['Previous Word']]
                        for prev_pos in prev_meaning:
                            if prev_pos in reconstruct and meaning in reconstruct[prev_pos]:
                                score += 1
                                break
                    if word in sentence_dict['Sentence']:
                        score -= 1
                    if word in pref['Dictionary']:
                        score += pref['Dictionary'][word][0] / 10
                    if len(word) <= pref['General']['Verbosity']:
                        score += 1
                    if word in reconstruct['Subject']:
                        score += 1

                    if score > max_score[1]:
                        max_score[0] = word
                        max_score[1] = score

                output += max_score[0] + ' '
                if dictionary[max_score[0]] in endings:
                    break
                sentence_dict['Sentence'] = output
                sentence_dict['Previous Word'] = max_score[0]

        return output

    def HMM(self, prev_speak=None, prev_responses=None):
        global quant
        model = None
        iterations = 10
        prev_responses_dict, reconstruct, MLR_model = {}, {}, {}

        if prev_responses:
            for response in prev_responses:
                prev_responses_dict[response] = Text(response).find()
            MLR_model = self.model_maker(prev_responses_dict, prev_responses)
            df = pd.DataFrame(MLR_model, columns=list(MLR_model.keys()))
            X = df.drop(['Score', 'Trials'], axis=1)
            Y = df[['Score']]
            model = sm.OLS(Y, X).fit()
            # Unfortunately, p-values can't be calculated with so few samples, so coefficients will be used instead
            max_coef = max(model.params)
            standard_coefs = abs(model.params / max_coef)

            for parameter in zip(standard_coefs, X):
                if 'Meaning' in parameter[1]:
                    index = int(re.findall(r'[\d]+', parameter[1])[0])
                    tag = quant['Meaning'][index]
                    category = 'Meaning'
                else:
                    tag = re.findall(r'[\w|]+', parameter[1])[0]
                    category = tag
                weights = [parameter[0], 1 - parameter[0]]
                flag = np.random.choice([1, 0], p=weights)
                if flag:
                    if category in quant:
                        if tag in reconstruct:
                            reconstruct[tag].append(quant[category][np.random.choice(X[parameter[1]])])
                        else:
                            reconstruct[tag] = [quant[category][np.random.choice(X[parameter[1]])]]
                    else:
                        if tag in reconstruct:
                            reconstruct[tag].append(np.random.choice(X[parameter[1]]))
                        else:
                            reconstruct[tag] = [np.random.choice(X[parameter[1]])]
                else:
                    if category in quant:
                        if tag in reconstruct:
                            reconstruct[tag].append(np.random.choice(quant[category]))
                        else:
                            reconstruct[tag] = [np.random.choice(quant[category])]
                    elif tag == 'Word Count':
                        if tag in reconstruct:
                            reconstruct[tag].append(random.randint(1, 20))
                        else:
                            reconstruct[tag] = [random.randint(1, 20)]
                    elif tag == 'Sentence Count':
                        if tag in reconstruct:
                            reconstruct[tag].append(random.randint(1, 4))
                        else:
                            reconstruct[tag] = [random.randint(1, 4)]
                    else:
                        if tag in reconstruct:
                            reconstruct[tag].append(random.uniform(0, 1))
                        else:
                            reconstruct[tag] = [random.uniform(0, 1)]

            reconstruct.pop('Constant')
            for parameter in reconstruct:
                if type(reconstruct[parameter][0]) == np.str_ or type(reconstruct[parameter][0]) == str:
                    reconstruct[parameter] = list(set(reconstruct[parameter]))
                else:
                    reconstruct[parameter] = sum(reconstruct[parameter]) / len(reconstruct[parameter])

        if prev_speak:
            prev_speak_dict = Text(prev_speak).find()
            for sentence in prev_speak_dict:
                for parameter in prev_speak_dict[sentence]:
                    if parameter == 'Mood':
                        if prev_speak_dict[sentence][parameter] == 'Interrogative':
                            if parameter in reconstruct:
                                reconstruct[parameter].append('Indicative')
                            else:
                                reconstruct[parameter] = ['Indicative']
                        elif prev_speak_dict[sentence][parameter] == 'Indicative':
                            if parameter in reconstruct:
                                reconstruct[parameter].append('Interrogative')
                            else:
                                reconstruct[parameter] = ['Interrogative']
                        else:
                            if parameter in reconstruct:
                                reconstruct[parameter].append(prev_speak_dict[sentence][parameter])
                            else:
                                reconstruct[parameter] = [prev_speak_dict[sentence][parameter]]
                    elif parameter == 'Subject':
                        if parameter in reconstruct:
                            reconstruct[parameter].append(prev_speak_dict[sentence][parameter])
                        else:
                            reconstruct[parameter] = [prev_speak_dict[sentence][parameter]]
                    elif parameter == 'Tense':
                        for tense in prev_speak_dict[sentence][parameter]:
                            if parameter in reconstruct:
                                reconstruct[parameter].append(prev_speak_dict[sentence][parameter][tense])
                            else:
                                reconstruct[parameter] = [prev_speak_dict[sentence][parameter][tense]]

        # Baseline, even if no phrase came before and nothing exists in database (blank slate)
        with open(f'characters/{self.character}.json', 'r') as f:
            pref = json.load(f)
        for parameter in pref['General']:
            if parameter not in reconstruct:
                reconstruct[parameter] = pref['General'][parameter]
        # Implementing SVO
        for parameter in quant['Meaning']:
            if parameter not in reconstruct:
                match parameter:
                    case 'CD':
                        reconstruct['CD'] = ['NN', 'NNP', 'NNS']
                    case 'DT':
                        reconstruct['DT'] = ['NN', 'NNP', 'NNS']
                    case 'JJ':
                        reconstruct['JJ'] = ['JJ', 'NN', 'NNP', 'NNS']
                    case 'JJR':
                        reconstruct['JJR'] = ['JJR', 'NN', 'NNP', 'NNS']
                    case 'JJS':
                        reconstruct['JJS'] = ['JJS', 'NN', 'NNP', 'NNS']
                    case 'NN':
                        reconstruct['NN'] = ['RB', 'RBR', 'RBS', 'POS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
                    case 'NNP':
                        reconstruct['NNP'] = ['RB', 'RBR', 'RBS', 'POS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
                    case 'NNS':
                        reconstruct['NNS'] = ['RB', 'RBR', 'RBS', 'POS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
                    case 'PDT':
                        reconstruct['PDT'] = ['NN', 'NNP', 'NNS']
                    case 'PRP':
                        reconstruct['PRP'] = ['CC', 'DT', 'IN', 'RP']
                    case 'PRP$':
                        reconstruct['PRP$'] = ['NN', 'NNP', 'NNS']
                    case 'RB':
                        reconstruct['RB'] = ['NN', 'NNP', 'NNS']
                    case 'RBR':
                        reconstruct['RBR'] = ['NN', 'NNP', 'NNS']
                    case 'RBS':
                        reconstruct['RBS'] = ['NN', 'NNP', 'NNS']
                    case 'TO':
                        reconstruct['TO'] = ['RB', 'RBR', 'RBS', 'POS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
                    case _:
                        pass

        # Add section to search the internet

        # RECONSTRUCTION
        if prev_responses:
            pref_words = []
            score, attempt_dict, check_dict = {}, {}, {}
            for i in range(iterations):
                # Need to do something to retain import words, maybe something with model.params somehow?
                attempt = self.reconstruction(reconstruct, pref_words)
                attempt_dict[attempt] = Text(attempt).find()
                attempt_model = self.model_maker(attempt_dict)
                for parameter in MLR_model:
                    if parameter not in attempt_model:
                        attempt_model[parameter] = MLR_model[parameter][0]
                a_copy = list(attempt_model)
                for parameter in a_copy:
                    if parameter not in MLR_model:
                        attempt_model.pop(parameter)
                attempt_model.pop('Score')
                attempt_model.pop('Trials')
                dfa = pd.DataFrame(attempt_model, columns=list(attempt_model.keys()))
                attempt_score = model.predict(dfa).iloc[0]
                score[attempt] = attempt_score
            output = max(score, key=score.get)
        else:
            output = self.reconstruction(reconstruct, None)

        # Updating character preferences
        output_words = Text(output).words()
        output_dict = Text(output).find()
        abbrev = {}
        verbosity = []
        for sentence in output_dict:
            for parameter in ['Sentiment', 'Emotion']:
                for elem in output_dict[sentence][parameter]:
                    tag = parameter + '|' + elem
                    output_dict[sentence][tag] = output_dict[sentence][parameter][elem]
        for word in output_words:
            verbosity.append(len(word))
        sent_count = len(output_dict)
        pref['General']['Sentence Count'] = len(output_dict)
        pref['General']['Verbosity'] = sum(verbosity) / len(verbosity)
        ignore = ['Count', 'Sentence Count', 'Verbosity']
        count = pref['General']['Count']
        for parameter in pref['General']:
            if parameter not in ignore:
                abbrev[parameter] = 0
                for sentence in output_dict:
                    abbrev[parameter] += output_dict[sentence][parameter]
                abbrev[parameter] = abbrev[parameter] / sent_count
                prev_parameter = pref['General'][parameter]
                if count:
                    updated_value = round(prev_parameter + ((abbrev[parameter] - prev_parameter) / count), 3)
                    pref['General'][parameter] = updated_value
                else:
                    pref['General'][parameter] = abbrev[parameter]
        count += 1

        with open(f'characters/{self.character}.json', 'w') as f:
            json.dump(pref, f)

        return output

    @staticmethod
    def reformat(phrase):
        non_spacing = [',', '-', ':']

        sentences = nltk.sent_tokenize(phrase)
        copy = ''
        i = 1
        for sentence in sentences:
            if i < len(sentences):
                sentence = sentence[0].upper() + sentence[1:] + ' '
            else:
                sentence = sentence[0].upper() + sentence[1:]
            copy += sentence
            for i in range(len(sentence)):
                if sentence[i] in non_spacing:
                    if i > 0 and sentence[i-1] == ' ':
                        copy = copy[:i-1] + copy[i:]
                    if i + 1 < len(sentence) and sentence[i+1] != ' ':
                        copy = copy[:i] + ' ' + copy[i:]
                    if i + 2 < len(sentence) and sentence[i+1] == ' ' and sentence[i+2].isdigit():
                        copy = copy[:i+1] + copy[i+2:]
            i += 1

        return copy

    # What if a character asks them to clarify what they said previously? I need a conversation memory.
    def speak(self, prev_speaker, prev_speak=None):
        with open(f'characters/{self.character}.json', 'r') as f:
            pref = json.load(f)

        # Repeat
        if prev_speaker == self.character:
            # Any Repeat
            if pref['Repeats']:
                if converse['Characters'] in pref['Repeats']:
                    output = self.HMM(None, pref['Repeats'][converse['Characters']])
                else:
                    score = {}
                    chars = converse['Characters'].split(' ')
                    for tag in pref['Repeats']:
                        tag_chars = tag.split(' ')
                        score[tag] = len(set(chars) & set(tag_chars))
                    closest = max(score, key=score.get)
                    output = self.HMM(None, pref['Repeats'][closest])
            # No Repeats
            else:
                output = self.HMM(None, None)

        # Response
        elif prev_speak:
            # Recorded Response
            if prev_speak in pref['Responses'][prev_speaker]:
                output = self.HMM(prev_speak, pref['Responses'][prev_speaker][prev_speak])
            # Any Response
            elif pref['Responses']:
                score = {}
                resp_words, prev_words = [], []
                for sentence in nltk.sent_tokenize(prev_speak):
                    prev_words += nltk.word_tokenize(sentence)
                for speaker in pref['Responses']:
                    for resp in pref['Responses'][speaker]:
                        for sentence in nltk.sent_tokenize(resp):
                            resp_words += nltk.word_tokenize(sentence)
                        similar = set(prev_words) & set(resp_words)
                        if resp in score:
                            score[resp] = max(score[resp], len(similar))
                        else:
                            score[resp] = len(similar)
                        resp_words = []

                closest = max(score, key=score.get)
                output = self.HMM(prev_speak, pref['Responses'][prev_speaker][closest])
            # No Responses
            else:
                output = self.HMM(prev_speak, None)

        # Greeting
        else:
            # Any Greeting
            if pref['Greetings']:
                if converse['Characters'] in pref['Greetings']:
                    output = self.HMM(None, pref['Greetings'][converse['Characters']])
                else:
                    score = {}
                    chars = converse['Characters'].split(' ')
                    for tag in pref['Greetings']:
                        tag_chars = tag.split(' ')
                        score[tag] = len(set(chars) & set(tag_chars))
                    closest = max(score, key=score.get)
                    output = self.HMM(None, pref['Greetings'][closest])
            # No Greetings
            else:
                output = self.HMM(None, None)

        return self.reformat(output)

    def score(self, speak, previous_speaker, previous_speak=None):
        global LW, CY, converse

        print(LW + f'Score for {self.character}: {speak} (eg. 0-10)')
        while True:
            try:
                score = int(input(CY))
                break
            except ValueError:
                print(LW + 'Please input a number from 1-10.')

        if not score:
            return

        with open(f'characters/{self.character}.json', 'r') as f:
            pref = json.load(f)
        for sentence in nltk.sent_tokenize(speak):
            sentence_meaning = nltk.pos_tag(sentence)
            for meaning in sentence_meaning:
                if meaning[0] in pref['Dictionary']:
                    prev_score = pref['Dictionary'][meaning[0]]
                    prev_score[1] += 1
                    prev_score[0] = round(prev_score[0] + ((score - prev_score[0]) / prev_score[1]), 3)
                else:
                    pref['Dictionary'][meaning[0]] = [score, 1]

        # This looks annoying and unnecessary, but it ensures every level of character's pref
        # (Greetings, Responses and Repeats) are checked properly if they exist
        if previous_speaker == self.character:
            if converse['Characters'] in pref['Repeats']:
                if speak in pref['Repeats'][converse['Characters']]:
                    prev_score = pref['Repeats'][converse['Characters']][speak]
                    prev_score[1] += 1
                    prev_score[0] = round(prev_score[0] + ((score - prev_score[0]) / prev_score[1]), 3)
                else:
                    pref['Repeats'][converse['Characters']][speak] = [score, 1]
            else:
                pref['Repeats'][converse['Characters']] = {speak: [score, 1]}
        elif previous_speak:
            if previous_speaker in pref['Responses']:
                if previous_speak in pref['Responses'][previous_speaker]:
                    if speak in pref['Responses'][previous_speaker][previous_speak]:
                        prev_score = pref['Responses'][previous_speaker][previous_speak][speak]
                        prev_score[1] += 1
                        prev_score[0] = round(prev_score[0] + ((score - prev_score[0]) / prev_score[1]), 3)
                    else:
                        pref['Responses'][previous_speaker][previous_speak][speak] = [score, 1]
                else:
                    pref['Responses'][previous_speaker][previous_speak] = {speak: [score, 1]}
            else:
                pref['Responses'][previous_speaker] = {previous_speak: {speak: [score, 1]}}
        else:
            if converse['Characters'] in pref['Greetings']:
                if speak in pref['Greetings'][converse['Characters']]:
                    prev_score = pref['Greetings'][converse['Characters']][speak]
                    prev_score[1] += 1
                    prev_score[0] = round(prev_score[0] + ((score - prev_score[0]) / prev_score[1]), 3)
                else:
                    pref['Greetings'][converse['Characters']][speak] = [score, 1]
            else:
                pref['Greetings'][converse['Characters']] = {speak: [score, 1]}

        with open(f'characters/{self.character}.json', 'w') as f:
            json.dump(pref, f)

    def driver(self):
        full = {}

        with open(f'characters/{self.character}_driver.txt', 'r') as f:
            drive = f.read()
        drive = drive.strip()
        responses = drive.split('||')
        for key_value in responses:
            key_value = key_value.strip()
            key_value = key_value.split(' : ')
            key = key_value[0].strip()
            values = key_value[1].split(' | ')
            full[key] = {}
            Text(key).find()
            for value in values:
                value = value.strip()
                Text(value).find()
                full[key][value] = [10, 1]

        with open(f'characters/characters.json', 'r') as f:
            characters = json.load(f)
        with open(f'characters/{self.character}.json', 'r') as f:
            pref = json.load(f)
        for character in characters:
            pref['Responses'][character] = full
        with open(f'characters/{self.character}.json', 'w') as f:
            json.dump(pref, f)


class App(threading.Thread):

    def __init__(self, user=None):
        self.user = user
        threading.Thread.__init__(self)
        self.start()

    def callback(self):
        self.root.quit()

    # Entry is needed for proper binding with 'return'
    def send(self, entry=None):
        global first_row, timer, reset_flag

        converse['Previous Speaker'] = self.user
        converse['Previous Speak'] = self.e.get()
        if first_row:
            self.txt.insert(tk.END, f'{self.user}: {self.e.get()}')
            first_row = False
        else:
            self.txt.insert(tk.END, f'\n{self.user}: {self.e.get()}')

        # Add in method to update parameters

        self.e.delete(0, tk.END)
        timer = 0
        reset_flag = True

    def insert(self, speak, speaker=None):
        global first_row

        if first_row:
            self.txt.insert(tk.END, f'{speaker}: {speak}')
            first_row = False
        else:
            self.txt.insert(tk.END, f'\n{speaker}: {speak}')

    def run(self):
        # Declarations down here to avoid Tkinter's multithreading issue.
        self.root = tk.Tk()  # noqa
        self.txt = tk.Text(self.root)  # noqa
        self.e = tk.Entry(self.root, width=100)  # noqa

        self.root.title('ARTITALK')
        self.root.bind('<Return>', self.send)
        self.root.protocol("WM_DELETE_WINDOW", self.callback)
        self.txt.grid(row=0, column=0, columnspan=2)
        self.e.grid(row=1, column=0)

        if self.user:
            tk.Button(self.root, text='Send', command=self.send).grid(row=1, column=1)

        self.root.mainloop()


# ---------------------------------------------------------------------------------------------------------------------
# MAINS


def non_user_main(non_users, app):
    global GR, converse, timer, training, reset_flag

    with open(f'characters/characters.json', 'r') as f:
        prefs = json.load(f)
    for character in non_users:
        time_mean = prefs[character]['Converse']['Time Mean']
        time_dev = prefs[character]['Converse']['Time Deviation']
        sel_time = round(max(0, np.random.normal(time_mean, time_dev)))
        converse['Time'][character] = sel_time

        silent_mean = prefs[character]['Converse']['Silent Mean']
        silent_dev = prefs[character]['Converse']['Silent Deviation']
        silence = round(max(0, np.random.normal(silent_mean, silent_dev)))
        converse['Silent'][character] = silence

        converse['Multiple'][character] = 0
        converse['Hold'][character] = {}

    # Remove this later
    holding = input('Holding: ')
    timer = 0
    reset_flag = False
    while True:
        print(timer, converse)
        for character in converse['Time']:
            attempts = converse['Multiple'][character]
            if character == converse['Previous Speaker']:
                if not converse['Hold'][character]:
                    attempts += 1
                # Send out any held onto lines
                copy = converse['Hold'][character]
                for line in list(copy):
                    if timer == copy[line]:
                        app.insert(line, character)
                        converse['Hold'][character].pop(line)
            else:
                converse['Hold'][character] = {}

            multiplier = prefs[character]['Converse']['Multiple'] * attempts
            if multiplier == 0:
                multiplier = 1
            char_timer = converse['Time'][character] * multiplier
            silence = converse['Silent'][character]
            if timer >= char_timer and attempts < silence:
                char = Character(character)
                speak = char.speak(converse['Previous Speaker'], converse['Previous Speak'])
                prev_sep_time = 0
                lines = nltk.sent_tokenize(speak)
                if len(lines) > 1:
                    app.insert(lines[0], character)
                    for line in lines[1:]:
                        sep_mean = prefs[character]['Converse']['Separation Mean']
                        sep_dev = prefs[character]['Converse']['Separation Deviation']
                        sep_time = round(max(0, np.random.normal(sep_mean, sep_dev))) + prev_sep_time
                        prev_sep_time = sep_time
                        converse['Hold'][character][line] = sep_time
                else:
                    app.insert(speak, character)
                    converse['Hold'][character] = {}
                timer = -1
                reset_flag = True
                if training:
                    char.score(speak, converse['Previous Speaker'], converse['Previous Speak'])
                converse['Previous Speaker'] = character
                converse['Previous Speak'] = speak
                break

        if reset_flag:
            # Reset speaking times
            for character in non_users:
                time_mean = prefs[character]['Converse']['Time Mean']
                time_dev = prefs[character]['Converse']['Time Deviation']
                sel_time = round(max(0, np.random.normal(time_mean, time_dev)))
                converse['Time'][character] = sel_time

                silent_mean = prefs[character]['Converse']['Silent Mean']
                silent_dev = prefs[character]['Converse']['Silent Deviation']
                silence = round(max(0, np.random.normal(silent_mean, silent_dev)))
                converse['Silent'][character] = silence

                converse['Multiple'][character] = 0
            reset_flag = False

        timer += 1
        time.sleep(1)

        # Add scoring parameters


# Allow for multiple characters to talk with each other
def main():
    global LW, training, converse
    user = None

    non_users = []
    characters = Character.select()
    converse['Characters'] = ' '.join(sorted(list(characters.keys())))
    for character in characters:
        if 'User' == characters[character]:
            user = character
        else:
            non_users.append(character)
    while True:
        print(LW + 'Train characters? (Yes/No)')
        answer = input().lower()
        if answer == 'yes':
            training = True
            break
        elif answer == 'no':
            training = False
            break
        else:
            print(LW + 'Unable to understand input. Please put in \'Yes\' or \'No\'')

    try:
        if user:
            app = App(user)
            non_user_main(non_users, app)
        else:
            app = App()
            non_user_main(non_users, app)
    except KeyboardInterrupt:
        print(LW + '\nClosing chatroom...')


if __name__ == '__main__':
    main()
