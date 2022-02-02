import re
import os
import json
import random
from string import ascii_lowercase
from colorama import Fore

# Third party modules (pandas, statsmodels, sklearn, word2number)
# import pandas as pd
# import statsmodels.api as sm
# from sklearn import linear_model
from word2number import w2n

# initializing global learning, which determines if character learns or not
learning = 0


# ----------------------------------------------------------------------------------------------------
# DATA MANAGEMENT


# INPUTS: N/A
# OUTPUTS: Adds JSON files for dictionaries and characters in dictionaries/ and characters/
# CALLS: N/A
def generate():
    if not os.path.exists('dictionaries'):
        os.makedirs('dictionaries')

    if not os.path.exists('characters'):
        os.makedirs('characters')

    if not os.path.exists('dictionaries/sym.json'):
        with open('dictionaries/sym.json', 'w') as f:
            json.dump({}, f)

    if not os.path.exists('characters/characters.json'):
        with open('characters/characters.json', 'w') as f:
            json.dump({'USERS': [], 'RESPONDERS': ['[1]', '[2]', '[3]', '[4]', '[5]']}, f)

    for i in range(5):
        if not os.path.exists(f'characters/[{i + 1}].json'):
            with open(f'characters/[{i + 1}].json', 'w') as f:
                json.dump({'Greetings': {}, 'Responses': {}}, f)

    for c in ascii_lowercase:
        if not os.path.exists(f'dictionaries/{c}.json'):
            with open(f'dictionaries/{c}.json', 'w') as f:
                json.dump({}, f)


# INPUTS: N/A
# OUTPUTS: Removes words from dictionary that have no definition
# CALLS: N/A
def clean():
    for c in ascii_lowercase:
        with open(f'dictionaries/{c}.json', 'r') as f:
            repo = json.load(f)
            for word in list(repo):
                if not repo[word]:
                    repo.pop(word, None)
        with open(f'dictionaries/{c}.json', 'w') as f:
            json.dump(repo, f)


# INPUTS: N/A
# OUTPUTS: Resorts dictionaries
# CALLS: N/A
def resort():
    for c in ascii_lowercase:
        with open(f'dictionaries/{c}.json', 'r') as f:
            repo = json.load(f)
        repo = dict(sorted(repo.items()))
        with open(f'dictionaries/{c}.json', 'w') as f:
            json.dump(repo, f)


# INPUTS: Variable to test for true emptiness
# OUTPUTS: Boolean, True if Empty, False if not
# CALLS: N/A
def empty(variable):
    if not variable or variable is None or variable == [''] or variable == [['']]:
        return True
    else:
        return False


# INPUTS: The definition and root of the word (where only one is not empty),
#        then the parameter name and value being tested against
# OUTPUTS: Boolean, True if dictionary parameter is empty when it shouldn't be,
#         and False if it should be empty or isn't empty
# CALLS: empty()
def logic(define, root, parameter, value):
    logic_one = ['Root', 'Derive', 'Meaning']
    logic_two = ['Meaning', 'Function', 'Synonyms', 'Antonyms', 'Root']

    if not empty(value):
        return False

    if empty(define) and empty(root):
        if parameter == 'Meaning' or parameter == 'Root':
            return True
    if not empty(define):
        if parameter not in logic_one:
            return True
    if not empty(root):
        if parameter not in logic_two:
            return True

    return False


# INPUTS: N/A
# OUTPUTS: Adds missing values to words in dictionaries
# CALLS: N/A
def info_add():
    # list_values = ['Derive', 'Synonyms', 'Antonyms'], just for later reference
    str_values = ['Root', 'Meaning', 'Function', 'Sentence']

    for c in ascii_lowercase:
        with open(f'dictionaries/{c}.json', 'r') as f:
            repo = json.load(f)
        for word in repo:
            for meaning in repo[word]:
                for parameter in repo[word][meaning]:
                    define = repo[word][meaning]['Meaning']
                    root = repo[word][meaning]['Root']
                    value = repo[word][meaning][parameter]
                    if logic(define, root, parameter, value):
                        if parameter in str_values:
                            print(Fore.LIGHTWHITE_EX + f'{word.lower()}: {define}')
                            print(Fore.LIGHTWHITE_EX + f'What is the {parameter.lower()} for {word.lower()}?')
                            value = input()
                        else:
                            print(Fore.LIGHTWHITE_EX + f'{word.lower()}: {define}')
                            print(Fore.LIGHTWHITE_EX + f'What are the {parameter.lower()} for {word.lower()}?')
                            value = input().split(' | ')

                        repo[word][meaning][parameter] = value
                        with open(Fore.LIGHTWHITE_EX + f'dictionaries/{c}.json', 'w') as f:
                            json.dump(repo, f)
                            print(Fore.LIGHTWHITE_EX + 'Updated!\n')


# INPUTS: A word
# OUTPUTS: The word with equal or fewer meanings (to save space)
# CALLS: Word.index()
def simplify(word):
    word_class = Word(word)
    index = word_class.index()

    with open(f'dictionaries/{index}.json', 'r') as f:
        repo = json.load(f)
    try:
        ins = repo[word.upper()]
        for meaning in ins:
            print(Fore.LIGHTWHITE_EX + f'{meaning}: {ins[meaning]["Meaning"]}')
        print(Fore.LIGHTWHITE_EX + f'{word} has {len(ins)} meanings. Which ones do you want? eg. 1 2 3')
        take = input().split()
        for meaning in ins:
            if int(meaning) not in take:
                ins.pop(meaning, None)
    except KeyError:
        return f'Unable to find "{word}"!'

    with open(Fore.LIGHTWHITE_EX + f'dictionaries/{index}.json', 'w') as f:
        repo.update(ins)
        json.dump(repo, f)

# ------------------------------------------------------------------------------------------------------
# CLASSES


# Handles the Word object, which is a string that has one or more linguistic meanings
class Word:
    def __init__(self, word):
        self.word = word

    # INPUTS: N/A
    # OUTPUTS: If math related, returns the dictionary definition with 'math' as a meaning.
    #         Returns False if not math related.
    # CALLS: Word.word
    def math(self):
        symbols = ['+', '-', '*', '/', '%', '^', '//']
        math = {'1': {
            'Root': None,
            'Derive': None,
            'Meaning': ['Math'],
            'Function': [],
            'Sentence': [],
            'Synonyms': [],
            'Antonyms': []
        }}

        try:
            number = w2n.word_to_num(self.word)
            if type(number) == int or type(number) == float:
                return math
        except ValueError:
            if self.word in symbols or self.word.isdigit():
                return math
            else:
                return False

    # INPUTS: N/A
    # OUTPUTS: Returns the first letter of the word, or 'sym', for dictionary accessing
    # CALLS: Word.word
    def index(self):
        index = self.word[0].lower()
        if not index.isalpha():
            index = 'sym'
        return index

    # INPUTS: N/A
    # OUTPUTS: Returns the definitions of a word from the larger dictionary,
    #         or raises a KeyError if not found or has no meanings in the larger dictionary
    # CALLS: Word.word
    def harvest(self):
        ins = {}
        index = self.word[0].upper()

        if index.isalpha():
            with open(f'data/D{index}.json', 'r') as f:
                repo = json.load(f)

            word_dict = repo[self.word.upper()]
            meanings = word_dict['MEANINGS']
            if not meanings:
                raise KeyError

            i = 0
            for key in meanings:
                ins[str(i + 1)] = {
                    'Root': None,
                    'Derive': None,
                    'Meaning': meanings[key][1],
                    'Function': meanings[key][0],
                    'Sentence': [],
                    'Synonyms': word_dict['SYNONYMS'],
                    'Antonyms': word_dict['ANTONYMS']
                }
                i += 1

        return ins

    # INPUTS: N/A
    # OUTPUTS: Inserts the definitions of the word into the main dictionary, if no definitions are found
    # CALLS: Word.word, Word.index(), Word.harvest(),
    def insert(self):
        meaning, function, sentence, synonyms, antonyms = [], [], [], [], []
        derive, root = None, None
        ins = {}
        index = self.index()

        print(Fore.LIGHTWHITE_EX + f'Does the word "{self.word}" derive from a root? (Yes/No)')
        if input().lower() == 'yes':
            print(Fore.LIGHTWHITE_EX + f'What is the root of the word?')
            root = input()
            print(Fore.LIGHTWHITE_EX + f'What kind of derivation is this?')
            derive = input()
            print(Fore.LIGHTWHITE_EX + f'Can you use the word in a sentence?')
            sentence.append(input())

            ins['1'] = {
                'Root': root,
                'Derive': derive,
                'Meaning': meaning,
                'Function': function,
                'Sentence': sentence,
                'Synonyms': synonyms,
                'Antonyms': antonyms
            }

        else:
            try:
                harvest = self.harvest()
                for meaning in harvest:
                    print(Fore.LIGHTWHITE_EX + f'{meaning}: {harvest[meaning]["Meaning"]}')
                print(Fore.LIGHTWHITE_EX + f'Would you like to use a harvested definition for {self.word}?')
                if input().lower() == 'yes':
                    if len(harvest) > 1:
                        print(f'{self.word} has {len(harvest)} meanings. Which ones do you want? eg. 1 2 3')
                        take = input().split()
                        for meaning in harvest:
                            if meaning in take:
                                ins[meaning] = harvest[meaning]
                    else:
                        ins = harvest
                else:
                    raise KeyError

            except KeyError:
                meaning = []
                while True:
                    print(Fore.LIGHTWHITE_EX + f'What is the meaning of the word \"{self.word}\"?')
                    meaning.append(input())
                    print(Fore.LIGHTWHITE_EX + 'What function does this word have in a sentence?')
                    function.append(input())
                    print(Fore.LIGHTWHITE_EX + 'Can you use the word in a sentence?')
                    sentence.append(input())
                    print(Fore.LIGHTWHITE_EX + 'Any synonyms for this word?')
                    synonyms.append(input().split(' | '))
                    print(Fore.LIGHTWHITE_EX + 'Any antonyms?')
                    antonyms.append(input().split(' | '))

                    for i in range(len(meaning)):
                        ins[str(i + 1)] = {
                            'Root': root,
                            'Derive': derive,
                            'Meaning': meaning[i],
                            'Function': function[i],
                            'Sentence': sentence[i],
                            'Synonyms': synonyms[i],
                            'Antonyms': antonyms[i]
                        }

                    print(Fore.LIGHTWHITE_EX + 'Are there any other meanings of this word? (Yes/No)')
                    if input().lower() != 'yes':
                        break

        with open(f'dictionaries/{index}.json', 'r') as f:
            repo = json.load(f)
        repo[self.word.upper()] = ins
        with open(f'dictionaries/{index}.json', 'w') as f:
            json.dump(repo, f)

    # INPUTS: N/A
    # OUTPUTS: Returns the meaning of the word from the main dictionary
    # CALLS: Word.word, Word.math(), Word.index(), Word.insert()
    def find(self):
        math = self.math()
        if self.math():
            return math

        index = self.index()
        with open(f'dictionaries/{index}.json', 'r') as f:
            repo = json.load(f)
        try:
            return repo[self.word.upper()]
        except KeyError:
            f.close()
            self.insert()
            with open(f'dictionaries/{index}.json', 'r') as f:
                repo = json.load(f)
                return repo[self.word.upper()]


# Handles the Sentence, which is a string of Words that give each other linguistic meaning when used a certain way
class Sentence:
    def __init__(self, sentence):
        self.sentence = sentence

    # INPUTS: A list of words, and the meanings of said words in a sentence
    # OUTPUTS: Returns the meaning of the sentence with an additional field denoting the mood of the sentence
    # CALLS: N/A
    @staticmethod
    def mood(words, sentence_dict):
        functions = []
        for i in range(len(words)):
            word = words[i]
            for meaning in sentence_dict[word]:
                functions += [sentence_dict[word][meaning]['Function']]

        # Indicative/Declarative (Fact) = Must end in period, can be any tense
        # Imperative (Command) = Must end in period, must use infinitives, negative commands use do+not+infinitive
        # Conditional (Condition) = Must have helping verb, typically has 'if' or 'when'
        # Subjunctive (Uncertainty) = Must have two simple present verbs, typically has 'that'
        # Interrogative (Question = Must have '?'

        if '?' in words:
            sentence_dict['Mood'] = 'Interrogative'

        elif 'do not' in words or 'Pronoun' not in functions:
            sentence_dict['Mood'] = 'Imperative'

        elif 'if' in words:
            sentence_dict['Mood'] = 'Conditional'

        elif 'that' in words:
            sentence_dict['Mood'] = 'Subjunctive'

        else:
            sentence_dict['Mood'] = 'Indicative'

        return sentence_dict

    # INPUTS: A list of words, and the meanings of said words in a sentence
    # OUTPUTS: Returns the meaning of the sentence with an additional field denoting the tense of the sentence
    # CALLS: N/A
    @staticmethod
    def tense(words, sentence_dict):
        tense, state = '', ''

        # This if-case checks that we aren't using interrogative phrasing,
        # where the subject would be slipped in between the two verbs
        if '?' in words:
            j = 2
        else:
            j = 1

        for i in range(len(words)):
            word = words[i]
            for meaning in sentence_dict[word]:
                function = sentence_dict[word][meaning]['Function']
                function = (f.lower() for f in function)
                derive = sentence_dict[words[i]][meaning]['Derive']
                if derive:
                    derive = derive.lower()
                if 'verb' in function:
                    if 'past' == derive:
                        tense = 'past'
                    elif 'future' == derive:
                        tense = 'future'
                    else:
                        tense = 'present'

                    # Checking to see if the state is not 'simple'
                    if words[i+j]:
                        function_second = sentence_dict[words[i+j]][meaning]['Function']
                        if 'verb' in (f.lower() for f in function_second):
                            # That means word[i] is an indicator of tense and word[i+1] is an indicator of state
                            derive_second = sentence_dict[words[i+j]][meaning]['Derive'].lower()
                            if 'gerund' == derive_second:
                                state = 'continuous'
                            elif word[i+j].lower() == 'been':
                                state = 'perfect continuous'
                            else:
                                state = 'perfect'
                            break
                        else:
                            state = 'simple'
                            break

                    else:
                        state = 'simple'
                        break
        if tense != '':
            tense = tense + ' ' + state
        sentence_dict['Tense'] = tense

        return sentence_dict

    # INPUTS: The meanings of each word in a sentence
    # OUTPUTS: Returns the meaning of the sentence with an additional field
    #          denoting the subjects and perspectives (first, second, third person) of the sentence
    # CALLS: N/A
    @staticmethod
    def subject(sentence_dict):
        subjects, perspectives = [], []

        for word in sentence_dict:
            for meaning in sentence_dict[word]:
                function = sentence_dict[word][meaning]['Function']
                derive = sentence_dict[word][meaning]['Derive']
                if derive:
                    derive = derive.lower()
                word_derive = [word.lower()] + [derive]
                if function == 'Pronoun' or function == 'Noun':
                    if 'i' in word_derive or 'we' in word_derive:
                        perspectives += ['First person']
                    if 'you' in word_derive:
                        perspectives += ['Second person']
                    else:
                        perspectives += ['Third person']
                    subjects += [word]

        sentence_dict['Subjects'] = subjects
        sentence_dict['Perspectives'] = perspectives

        return sentence_dict

    # INPUTS: N/A
    # OUTPUTS: Returns a list of the words in a sentence, splitting up contractions ('s, 'm, n't) as well
    # CALLS: Sentence.sentence
    def words(self):
        s = self.sentence

        for i in range(len(s)):
            if s[i] == "\'":
                if s[i - 1] == 'n' and s[i + 1] == 't':
                    # Can't is the only one that splits into "ca" and "n't", so change to "Cann't"
                    if s[i - 3:i].lower() == 'can':
                        s = s[:i] + "~n\'t" + s[i + 2:]
                        i += 3
                    else:
                        s = s[:i - 1] + "~n\'t" + s[i + 2:]
                        i += 3
                else:
                    s = s[:i] + "~\'" + s[i + 1:]
                    i += 2
            else:
                i += 1

        words = re.findall(r"[\w']+|[\s]|[.,!?;\s]", s)
        return words

    # INPUTS: N/A
    # OUTPUTS: Returns the meanings of all the words and context in a sentence as a dictionary
    # CALLS: Sentence.words(), Sentence.subject(), Sentence.tense(), Sentence.mood()
    def find(self):
        sentence_dict = {}
        words = self.words()

        for word in words:
            word_dict = Word(word)
            sentence_dict[word] = word_dict.find()

        # Adds additional parameters
        sentence_dict = self.subject(sentence_dict)
        sentence_dict = self.tense(words, sentence_dict)
        sentence_dict = self.mood(words, sentence_dict)

        return sentence_dict


# Handles the Character object, which is a person (user or responder) that generates sentences with linguistic
# meaning, according to their own preferences
# NOTE: The HMM model will replace these Character functions:
# weighted_selection(), change(), first_state(), second_state(), third_state(),
# since they all use phrases, not singular words for constructing the sentence
class Character:
    def __init__(self, character):
        self.character = character

    # INPUTS: A role (user or responder)
    # OUTPUTS: Returns the people that are identified with this role
    # CALLS: N/A
    @staticmethod
    def all(role):
        with open('characters/characters.json', 'r') as f:
            characters = json.load(f)
        people = characters[role]
        return people

    # INPUTS: A role
    # OUTPUTS: Adds the role to a JSON file if not already there, returning a string of the action taken
    # CALLS: Character.character
    def insert(self, role):
        with open('characters/characters.json', 'r') as f:
            characters = json.load(f)
        people = characters[role]
        if self.character not in people:
            people += [self.character]
            with open('characters/characters.json', 'w') as f:
                json.dump(characters, f)
            if role == 'USERS':
                for character in characters['RESPONDERS']:
                    with open(f'characters/{character}.json', 'r') as f:
                        pref = json.load(f)
                    for key in pref:
                        pref[key][self.character] = {}
                    with open(f'characters/{character}.json', 'w') as f:
                        json.dump(pref, f)
                return f'{self.character} added to repository!'
            if role == 'RESPONDERS':
                with open(f'characters/{self.character}.json', 'w') as f:
                    json.dump({'Greetings': {}, 'Responses': {}}, f)
        else:
            return f'{self.character} already in repository!'

    # INPUTS: N/A
    # OUTPUTS: Determines if the character will learn new words or not
    # CALLS: Character.character
    def learning(self):
        global learning

        print(Fore.LIGHTWHITE_EX + f'Do you want {self.character} to learn during this session? (Yes/No)')
        answer = input()
        if answer.lower() == 'yes':
            learning = 1
        else:
            learning = 0

    # INPUTS: A dictionary of preferences
    # OUTPUTS: Returns a preferred phrase as a greeting or response, selected by weighted probability
    # CALLS: N/A
    @staticmethod
    def weighted_selection(pref):
        global learning
        keys, values, trials = [], [], []

        for key in pref:
            keys += [key]
            values += [pref[key][0]]
            trials += [pref[key][1]]

        # Determines if we try a different phrase that hasn't been recorded.
        # This gives us more samples to perform the HMM on later.
        avg_value = sum(values) / len(values)
        avg_trial = sum(trials) / (1 + sum(trials))
        avg_weight = avg_value * avg_trial
        gate = random.choices([1, 0], [avg_weight, 1 - avg_weight], k=1)[0]
        if gate == 0 and learning == 1:
            values = [0.0]

        phrase = random.choices(keys, values, k=1)[0]
        return phrase

    # INPUTS: A user, the user's input, and a dictionary of preferences
    # OUTPUTS: Returns the preferred phrase, using second and third_state for edge cases
    # CALLS: Character.character, Character.weighted_selection(), Character.second_state(), Character.third_state()
    def first_state(self, user, user_input, pref):
        try:
            responses = pref['Responses'][user][user_input.lower()]
            return self.weighted_selection(responses)

        except (ValueError, ZeroDivisionError):
            return self.second_state(user, user_input, pref)

        except KeyError:
            return self.third_state(user, user_input, pref)

    # INPUTS: A user, the user's input, and a dictionary of preferences
    # OUTPUTS: Returns any response to any input
    # CALLS: N/A
    @staticmethod
    def second_state(user, user_input, pref):
        psb = []
        responses = pref['Responses'][user]

        for string in responses:
            if string != user_input.lower():
                psb += [string]
        if psb:
            response = psb[random.randint(0, len(psb) - 1)]
        else:
            response = user_input

        return response

    # INPUTS: A user, the user's input, and a dictionary of preferences
    # OUTPUTS: Returns any input to any input
    # CALLS: N/A
    @staticmethod
    def third_state(user, user_input, pref):
        keys, values = [], []
        score = {}

        # Tries first to find another
        try:
            responses = pref['Responses'][user]
            for string in responses:
                score[string] = 0
                words = re.findall(r"[\w']+|[.,!?;]", string)
                for word in words:
                    if word in user_input.lower():
                        score[string] += 1

            closest = max(score, key=score.get)
            response = responses[closest]
            for key in response:
                keys += [key]
                values += [response[key][0]]
            if len(values) == 1 and values[0] == 0:
                values[0] = 1.0
            response = random.choices(keys, values, k=1)[0]
            return response
        except (ValueError, KeyError):
            return user_input

    # INPUTS: A phrase (a greeting or response from the responder)
    # OUTPUTS: Returns the phrase correctly capitalized
    # CALLS: Word.word, Word.index()
    @staticmethod
    def reformat(phrase):
        phrase = phrase.lower()
        personal_pronouns = ['you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']

        sentences = re.findall(r"[\w'\s+,-]+|[.!?;\s]+", phrase)
        phrase = ''

        for sentence in sentences:
            sentence = Sentence(sentence)
            words = sentence.words()
            for word in words:
                if word not in personal_pronouns:
                    word_dict = Word(word)
                    index = word_dict.index()
                    with open(f'dictionaries/{index}.json', 'r') as f:
                        repo = json.load(f)
                    for meaning in repo[word.upper()]:
                        function = repo[word.upper()][meaning]['Function']
                        if function and function.lower() == 'pronoun':
                            word = word[0].upper() + word[1:]
                phrase += word

        sentences = re.findall(r"[\w'\s,]+|[.!?;\s]+", phrase)
        phrase = ''

        for sentence in sentences:
            sentence = sentence[0].upper() + sentence[1:]
            phrase += sentence

        # Have to clean up the "can't" issue created from the sentence.words() contraction issue
        phrase = phrase.replace('cann\'t', 'can\'t')
        phrase = phrase.replace('Cann\'t', 'Can\'t')

        return phrase

    # INPUTS: A phrase, and the phrase's meanings.
    # OUTPUTS: Returns the phrase with the possibility of replacing words with synonyms
    # CALLS: Sentence.sentence, Sentence.words()
    @staticmethod
    def change(phrase, phrase_meaning):
        sentence = Sentence(phrase)
        words = sentence.words()
        phrase = ''

        for word in words:
            try:
                synonyms = phrase_meaning[word]['Synonyms']
                words = [word] + synonyms
                phrase += random.choice(words)
            except KeyError:
                phrase += word

        return phrase

    # INPUTS: A user(name)
    # OUTPUTS: Returns the preferred greeting to a particular user
    # CALLS: Character.character, Character.change(), Character.weighted_selection(), Character.reformat()
    def greeting(self, user):
        greeting_dict = {}
        with open(f'characters/{self.character}.json', 'r') as f:
            pref = json.load(f)
        try:
            greetings = pref['Greetings'][user]
            greeting = self.weighted_selection(greetings)
            sentences = re.findall(r"[\w'\s]+|[.,!?;\s]+", greeting)
            for sentence in sentences:
                sentence_dict = Sentence(sentence)
                greeting_dict.update(sentence_dict.find())
            greeting = self.change(greeting, greeting_dict)
            greeting = self.reformat(greeting)
            return greeting
        except (ValueError, ZeroDivisionError):
            try:
                responses = pref['Responses'][user]
                greeting = self.weighted_selection(responses)
                greeting = self.reformat(greeting)
                return greeting
            except KeyError:
                return

    # INPUTS: A user, and the user's input
    # OUTPUTS: Returns the character's preferred response to a user's input
    # CALLS: Character.character, Character.first_state(), Character.change(), Character.reformat()
    def response(self, user, user_input):
        user_meaning, response_meaning = {}, {}
        user_sentences = re.findall(r"[\w'\s-]+|[.,!?;\s]+", user_input)

        # Finds the meaning of the user input
        for user_sentence in user_sentences:
            user_sentence_dict = Sentence(user_sentence)
            user_meaning.update(user_sentence_dict.find())

        with open(f'characters/{self.character}.json', 'r') as f:
            pref = json.load(f)
        response = self.first_state(user, user_input, pref)
        response_sentences = re.findall(r"[\w'\s-]+|[.,!?;\s]+", response)
        for response_sentence in response_sentences:
            response_sentence_dict = Sentence(response_sentence)
            response_meaning.update(response_sentence_dict.find())

        response = self.change(response, response_meaning)
        response = self.reformat(response)
        return response

    # INPUTS: A user, the user's input, a response,
    #         and the corresponding score (of how well the response addressed the input)
    # OUTPUTS: Adds the score to the character's preferences under the correct interaction
    # CALLS: Character.character
    def scoring(self, user, user_input, response, score):
        score = int(score) / 10

        with open(f'characters/{self.character}.json', 'r') as f:
            pref = json.load(f)

        if user_input:
            try:
                prev_score = pref['Responses'][user][user_input.upper()][response]
                prev_score[1] += 1
                # Calculating new average score
                prev_score[0] = round(prev_score[0] + ((score-prev_score[0])/prev_score[1]), 3)
            except KeyError:
                pref['Responses'][user][user_input.upper()] = {response: [score, 1]}
        else:
            try:
                prev_score = pref['Greetings'][user][response.upper()]
                prev_score[1] += 1
                prev_score[0] = round(prev_score[0] + ((score-prev_score[0])/prev_score[1]), 3)
            except KeyError:
                pref['Greetings'][user][response.upper()] = [score, 1]

        with open(f'characters/{self.character}.json', 'w') as f:
            json.dump(pref, f)

    # INPUTS: N/A
    # OUTPUTS: Adds sample interactions between a user and a responder
    # CALLS: Character.character
    def driver(self):
        full = {}

        with open(f'characters/{self.character}_driver.txt', 'r') as f:
            pref = f.read()
            pref = pref.strip()
            responses = pref.split('||')
            for key_value in responses:
                key_value = key_value.split(' : ')
                key = key_value[0].strip()
                values = key_value[1].split(' | ')
                full[key.upper()] = {}
                for value in values:
                    value = value.strip()
                    full[key.upper()].update({value: [1.0, 1]})

        with open(f'characters/{self.character}.json', 'r') as f:
            pref = json.load(f)
        users = pref['Responses']
        for user in users:
            pref['Responses'][user].update(full)
        with open(f'characters/{self.character}.json', 'w') as f:
            json.dump(pref, f)

# ----------------------------------------------------------------------------------------------------
# MAIN


# Handles the chatbot AI
# CALLS: generate(), clean(), resort(), Character.all(), Character.character, Character.insert(),
#        Character.learning(), Character.greeting(), Character.scoring(), Character.response()
def main():
    global learning
    generate()
    clean()
    resort()

    try:
        roles = ['USERS', 'RESPONDERS']
        examples = {
            'USERS': ['[ira]', 'Create new user...'],
            'RESPONDERS': ['[1]', 'Create new responder...']
        }
        people = {}

        # Handles multiple users and character selection
        for role in roles:
            while True:
                characters = Character.all(role)
                ref_role = role.lower()[:(len(role) - 1)]
                characters += [f'Create new {ref_role}...']
                i = 1
                for character in characters:
                    print(f'{i}: {character}')
                    i += 1
                print(f'Please select a {ref_role}. (eg. {examples[role][0]}, {examples[role][1]})')
                person = input()
                if 'Create new' in person:
                    print('Please insert new name...')
                    new_person = input()
                    person_char = Character(new_person)
                    print(person_char.insert(role))
                    people[ref_role] = new_person
                    break
                elif person not in characters:
                    print(f'Unable to find {ref_role}! Please try again...\n')
                else:
                    people[ref_role] = person
                    break

        user = people['user']
        char = people['responder']
        character = Character(char)
        character.learning()
        greeting = character.greeting(user)
        if greeting:
            print(Fore.LIGHTGREEN_EX + f'{char}: {greeting}')
            if learning == 1:
                # Scoring of Greeting
                print(Fore.LIGHTWHITE_EX + 'Scoring? (0-10): ')
                character.scoring(user, None, greeting, input())

        while True:
            # User input
            print(Fore.CYAN + 'User: ')
            user_input = input()

            # Response to User Input
            response = character.response(user, user_input)
            print(Fore.LIGHTGREEN_EX + f'{char}: {response}')

            if learning == 1:
                # Scoring of Response
                print(Fore.LIGHTWHITE_EX + 'Scoring? (0-10): ')
                character.scoring(user, user_input, response, input())

    except KeyboardInterrupt:
        print('\nClosing...')


if __name__ == "__main__":
    main()
