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


# Generates new dictionaries for characters, symbols and words starting with each letter in new directories
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


# Cleans dictionaries of any words that have no definitions
def clean():
    for c in ascii_lowercase:
        with open(f'dictionaries/{c}.json', 'r') as f:
            repo = json.load(f)
            for word in list(repo):
                if not repo[word]:
                    repo.pop(word, None)
        with open(f'dictionaries/{c}.json', 'w') as f:
            json.dump(repo, f)


# Resorts the dictionaries by key
def resort():
    for c in ascii_lowercase:
        with open(f'dictionaries/{c}.json', 'r') as f:
            repo = json.load(f)
        repo = dict(sorted(repo.items()))
        with open(f'dictionaries/{c}.json', 'w') as f:
            json.dump(repo, f)


# Expanded empty conditions
def empty(variable):
    if not variable or variable is None or variable == [''] or variable == [['']]:
        return True
    else:
        return False


# Using logic to save time in adding information
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


# Adds info to words in dictionaries
def info_add():
    # list_values = ['Derive', 'Synonyms', 'Antonyms'], just for reference
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


# Finds a word in the dictionary and asks for which definitions to keep
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


# Handles the word
class Word:
    def __init__(self, word):
        self.word = word

    # Figures out if the word is actually math related, then handles that separately
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

    # Finds the first letter of the word, return 'sym' if it's a symbol    
    def index(self):
        index = self.word[0].lower()
        if not index.isalpha():
            index = 'sym'
        return index

    # Looks for the word in another dictionary, inserting and returning 1 that if found
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

    # Inserts the word's meaning into the dictionary
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

    # Finds the meaning of the word            
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


# Handles the sentence
class Sentence:
    def __init__(self, sentence):
        self.sentence = sentence

    # Interprets the mood of the sentence and adds it to the sentence dictionary
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
            return sentence_dict

        elif 'do not' in words or 'Pronoun' not in functions:
            sentence_dict['Mood'] = 'Imperative'
            return sentence_dict

        elif 'if' in words:
            sentence_dict['Mood'] = 'Conditional'

        elif 'that' in words:
            sentence_dict['Mood'] = 'Subjunctive'

        else:
            sentence_dict['Mood'] = 'Indicative'

        return sentence_dict

    # Interprets the tense of the sentence and adds it to the sentence dictionary
    @staticmethod
    def tense(words, sentence_dict):
        tense, state = '', ''

        # This if-case checks that we aren't using interrogative phrasing,
        # where the word 'you' would be slipped in between
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

    # Finds and adds the subjects and perspectives in a sentence to the sentence dictionary.
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

    # Find words in a sentence, also splitting up contractions like 'm, 's and n't
    def words(self):
        s = self.sentence

        for i in range(len(s)):
            if s[i] == "\'":
                if s[i - 1] == 'n' and s[i + 1] == 't':
                    # Can't is the only one that splits into "ca" and "n't"
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

    # Finds the meaning of the words in the sentence
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


# Handles the character, or the personalized weights of each person in the database.
class Character:
    def __init__(self, character):
        self.character = character

    # Finds characters of a specific role (user or responder) in repository
    @staticmethod
    def characters(role):
        with open('characters/characters.json', 'r') as f:
            characters = json.load(f)
        people = characters[role]
        return people

    # Inserts character of a specific role into repository
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

    # Determines if the character will learn this session
    def learning(self):
        global learning

        print(Fore.LIGHTWHITE_EX + f'Do you want {self.character} to learn during this session? (Yes/No)')
        answer = input()
        if answer.lower() == 'yes':
            learning = 1
        else:
            learning = 0

    # Determines what phrase a character would prefer from a list of phrases
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

    # Will reformat the sentences correctly.
    @staticmethod
    def reformat(response, response_meaning):
        response = response.lower()
        personal_pronouns = ['you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']

        sentences = re.findall(r"[\w'\s+,+-]+|[.!?;\s]+", response)
        response = ''

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
                response += word

        sentences = re.findall(r"[\w'\s,]+|[.!?;\s]+", response)
        response = ''

        for sentence in sentences:
            sentence = sentence[0].upper() + sentence[1:]
            response += sentence

        # Have to clean up the "can't" issue created from the sentence.words() contraction issue
        response = response.replace('cann\'t', 'can\'t')
        response = response.replace('Cann\'t', 'Can\'t')

        return response

    # Determines if the character will greet the user, and how
    def greeting(self, user):
        with open(f'characters/{self.character}.json', 'r') as f:
            pref = json.load(f)
        try:
            greetings = pref['Greetings'][user]
            greeting = self.weighted_selection(greetings)
            sentences = re.findall(r"[\w'\s]+|[.,!?;\s]+", greeting)
            for sentence in sentences:
                sentence_dict = Sentence(sentence)
                sentence_dict.find()
            greeting = self.reformat(greeting, greetings)
            return greeting
        except (ValueError, ZeroDivisionError):
            try:
                responses = pref['Responses'][user]
                greeting = self.weighted_selection(responses)
                greeting = self.reformat(greeting, responses)
                return greeting
            except KeyError:
                return

    # Handles how the character will use synonyms in their response
    @staticmethod
    def change(response, response_meaning):
        change = ''
        sentence = Sentence(response)
        words = sentence.words()
        print(words)

        for word in words:
            try:
                synonyms = response_meaning[word]['Synonyms']
                words = [word] + synonyms
                change += random.choice(words)
            except KeyError:
                change += word

        return change

    # Change out first, second and third state for HMM model (not looking for phrases anymore)
    # Returns the best response to input
    def first_state(self, user, user_input, pref):
        try:
            responses = pref['Responses'][user][user_input.lower()]
            return self.weighted_selection(responses)

        except (ValueError, ZeroDivisionError):
            return self.second_state(user, user_input, pref)

        except KeyError:
            return self.third_state(user, user_input, pref)

    # Returns different response to input
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

    # Returns any input to input
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

    # Handles how the character should respond to the user's input.   
    def response(self, user, user_input):
        user_meaning, response_meaning = {}, {}
        user_sentences = re.findall(r"[\w'\s]+|[.,!?;\s]+", user_input)

        # Finds the meaning of the user input
        for user_sentence in user_sentences:
            user_sentence_dict = Sentence(user_sentence)
            user_meaning.update(user_sentence_dict.find())

        with open(f'characters/{self.character}.json', 'r') as f:
            pref = json.load(f)
        response = self.first_state(user, user_input, pref)
        response_sentences = re.findall(r"[\w'\s]+|[.,!?;\s]+", response)
        for response_sentence in response_sentences:
            response_sentence_dict = Sentence(response_sentence)
            response_meaning.update(response_sentence_dict.find())

        response = self.change(response, response_meaning)
        response = self.reformat(response, response_meaning)
        return response

    # Handles scoring of response to user's input.
    # Add more scoring details to check for other parameters and score more accurately. 
    def scoring(self, user, user_input, response, score):
        score = int(score) / 10

        with open(f'characters/{self.character}.json', 'r') as f:
            pref = json.load(f)

        if user_input:
            try:
                prev_score = pref['Responses'][user][user_input.upper()][response]
                prev_score[1] += 1
                prev_score[0] = round((score + prev_score[0]) / prev_score[1], 3)
            except KeyError:
                pref['Responses'][user][user_input.upper()] = {response: [score, 1]}
        else:
            try:
                prev_score = pref['Greetings'][user][response.upper()]
                prev_score[1] += 1
                prev_score[0] = round((score + prev_score[0]) / prev_score[1], 3)
            except KeyError:
                pref['Greetings'][user][response.upper()] = [score, 1]

        with open(f'characters/{self.character}.json', 'w') as f:
            json.dump(pref, f)

    # Adds some sample responses to character.
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


# Handles the allocation of user inputs and responses.
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
                characters = Character.characters(role)
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
        character.driver()
        greeting = character.greeting(user)
        if greeting:
            print(Fore.LIGHTGREEN_EX + f'{char}: {greeting}')
            if learning == 1:
                # Scoring of Greeting
                print(Fore.LIGHTWHITE_EX + 'Scoring? (0-10): ')
                character.scoring(None, user, greeting, input())

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
        print('Closing...')


if __name__ == "__main__":
    main()
