import sys
import math
import os
import decimal
from scipy import spatial, stats
import numpy as np


movie_list = []
inv_ratings_movie = {}

class movie:

    def __init__(self, input_id, input_rating_list, input_ratings, train_test_input):

        self.average_rating = 1
        self.euclidean_length = 1
        self.pearson_length = 1
        self.train_test = train_test_input
        
        if self.train_test == 0:
            self.id = str(input_id)
            self.ratings = {}
            
            for index, rating in enumerate(input_rating_list):
                self.ratings[str(index+1)] = int(rating)

        elif self.train_test == 1:
            self.id = str(input_id)
            self.targets = [] 
            self.ratings = input_ratings 
            for key in input_ratings:
                if input_ratings[key] == 0:
                    self.targets.append(key)
                    
    def get_rating(self, user_id):

        if user_id in self.ratings:
            return int(self.ratings[user_id])

        return 0;

    def calc_euclidean_length(self):
        total = 0
        for key in self.ratings:
            if int(self.ratings[key]) != 0:
                total = total + int(self.ratings[key])**2

        return decimal.Decimal(total).sqrt()

    def calc_pearson_length(self):
        total = 0
        for key in self.ratings:
            if int(self.ratings[key]) != 0:
                total = total + ((int(self.ratings[key]) - self.average_rating))**2
                
        return decimal.Decimal(total).sqrt()

    def calc_average_rating(self): 
        count = 0
        total = 0
        for key in self.ratings:
            if int(self.ratings[key]) != 0:
                count += 1
                total += int(self.ratings[key])

        return (total/count)

    def add_rating(self, user_id, rating):
        self.ratings[user_id] = int(rating)
        if rating == 0 and self.train_test == 1:
            self.targets.append(user_id)

            
                        
class user:

    def __init__(self, input_id, input_rating_list, input_ratings, train_test_input):

        self.average_rating = 1
        self.euclidean_length = 1
        self.pearson_length = 1
        self.train_test = train_test_input
        
        if self.train_test == 0:
            self.id = str(input_id)
            self.ratings = {}
            
            for index, rating in enumerate(input_rating_list):
                self.ratings[str(index+1)] = int(rating)

        elif self.train_test == 1:
            self.id = str(input_id)
            self.targets = [] 
            self.ratings = input_ratings 
            for key in input_ratings:
                if input_ratings[key] == 0:
                    self.targets.append(key)
                    
    def get_rating(self, movie_id):

        if movie_id in self.ratings:
            return int(self.ratings[movie_id])

        return 0;

    def calc_euclidean_length(self):
        total = 0
        for key in self.ratings:
            if int(self.ratings[key]) != 0:
                total = total + int(self.ratings[key])**2

        return decimal.Decimal(total).sqrt()

    def calc_pearson_length(self):
        total = 0
        for key in self.ratings:
            if int(self.ratings[key]) != 0:
                total = total + (int(self.ratings[key]) - self.average_rating)**2
                
        return decimal.Decimal(total).sqrt()

    def calc_average_rating(self): 
        count = 0
        total = 0
        for key in self.ratings:
            if int(self.ratings[key]) != 0:
                count += 1
                total += int(self.ratings[key])

        return (total/count)

    def add_rating(self, movie_id, rating):
        self.ratings[movie_id] = int(rating)
        if rating == 0 and self.train_test == 1:
            self.targets.append(movie_id)

            

def set_characteristics(list_user_input):
    list_user = list_user_input
    for user_id in list_user:
        user_id.average_rating = user_id.calc_average_rating()
        user_id.euclidean_length = user_id.calc_euclidean_length()
        user_id.pearson_length = user_id.calc_pearson_length()

    return list_user

def check_common(test_user, train_user):
    count = 0
    for movie_id in test_user.ratings:
        for movie_id2 in train_user.ratings:
            if movie_id == movie_id2:
                count+=1
            if count == 2:
                break
    return count

def calculate_cosine_similarity(test_user, train_user):
    test_list = []
    train_list = []
    for movie_id in test_user.ratings:
        if movie_id in train_user.ratings:
            if int(test_user.ratings[movie_id]) != 0 and int(train_user.ratings[movie_id]) != 0:
                test_list.append(int(test_user.ratings[movie_id]))
                train_list.append(int(train_user.ratings[movie_id]))

    if len(test_list) >= 2:
        result =  1 - spatial.distance.cosine(test_list, train_list)
    else:
        result = 0

    if math.isnan(result):
        return 0
    else:
        return result
        

def calculate_cosine_prediction(test_user_input, train_list_input, inv_ratings_input):
    test_user = test_user_input
    train_list = train_list_input
    inv_ratings = inv_ratings_input
    processed_user = test_user
    for target in test_user.targets:
        #print("Test User ID: ", test_user.id)
        if target in inv_ratings:
            n = 90
            top_n = {}
            total_similarity = 0
            predict_rating = 0
            #Select and get top k similar users
            for trainer_id in inv_ratings[target]:
                
                if (len(inv_ratings[target]) == 0):
                    print("There is no rating data for movie ", target)


                similarity = calculate_cosine_similarity(test_user, train_list[int(trainer_id)-1])
                if train_list[int(trainer_id)-1].ratings[target] > 5:
                    print("User ID: {}, Rating: {}, Similarity: {}".format(trainer_id, train_list[int(trainer_id)-1].ratings[target], similarity))

    
                #print("Movie ID: {}, Test User ID: {}, Trainer User ID: {}, Similarity: {}".format(target, test_user.id, trainer_id, similarity))
                if len(top_n) < n:
                    top_n[trainer_id] = similarity
                else:
                    top_n[trainer_id] = similarity
                    min_key = min(top_n, key=top_n.get)
                    top_n.pop(min_key)


            #compute summation of w (denominator):
            count = 0
            for trainer_id in top_n:
                total_similarity += top_n[trainer_id]
                count+=1
                #print("Count: {}, Movie ID: {}, Test User ID: {}, Trainer User ID: {}, Similarity: {}".format(count, target, test_user.id, trainer_id, top_n[trainer_id]))


            #compute summation of product of u and r (numerator):
            for trainer_id in top_n:
                if total_similarity != 0:
                    predict_rating += (top_n[trainer_id]*train_list[int(trainer_id)-1].ratings[target])
                    #print("Weight: {}, Rating: {}".format(top_n[trainer_id], train_list[int(trainer_id)-1].ratings[target]))

            if total_similarity != 0:
                predict_rating = predict_rating/total_similarity
            elif total_similarity == 0: # ASSUME THAT WITHOUT OTHER DATA, THE USER GIVES THE AVG RATING
                predict_rating = test_user.average_rating
                print(test_user.average_rating)

            #print("Movie ID: {}, Test User ID: {}, Trainer User ID: {}, Predicted Rating: {}".format(target, test_user.id, trainer_id, predict_rating))

            processed_user.ratings[target] = int(round(predict_rating))

    return processed_user


def calculate_pearson_correlation(test_user, train_user):
    test_list = []
    train_list = []
    numerator = 0
    for movie_id in test_user.ratings:
        if movie_id in train_user.ratings:
            if int(test_user.ratings[movie_id]) != 0 and int(train_user.ratings[movie_id]) != 0:
                test_list.append(int(test_user.ratings[movie_id]))
                train_list.append(int(train_user.ratings[movie_id]))
                #print(train_user.ratings[movie_id])

    if len(test_list) >= 2:
        array_test_list = np.asarray(test_list)
        array_train_list = np.asarray(test_list)
        result = pearson_correlation(array_test_list, array_train_list, test_user, train_user)
    else:
        result = 0

    return result


def pearson_correlation(numbers_x, numbers_y, test_user, train_user):
    x = numbers_x - test_user.average_rating
    y = numbers_y - train_user.average_rating
    denom = float(test_user.pearson_length * train_user.pearson_length)
    if denom != 0:
        return (x * y).sum() / float(test_user.pearson_length * train_user.pearson_length)
    else:
        return 0

def calculate_pearson_prediction(test_user_input, train_list_input, inv_ratings_input):
    test_user = test_user_input
    train_list = train_list_input
    inv_ratings = inv_ratings_input
    processed_user = test_user

    for target in test_user.targets:
        if target in inv_ratings:
            n = 75
            top_n = {}
            total_similarity = 0
            min_key = str()
            for trainer_id in inv_ratings[target]:
                if (len(inv_ratings[target]) == 0):
                    print("There is no rating data for movie ", target)
                    
                similarity = calculate_pearson_correlation(test_user, train_list[int(trainer_id)-1])
                if train_list[int(trainer_id)-1].ratings[target] > 5:
                    print("User ID: {}, Rating: {}, Similarity: {}".format(trainer_id, train_list[int(trainer_id)-1].ratings[target], similarity))
                if len(top_n) < n:
                    top_n[trainer_id] = similarity
                else:
                    top_n[trainer_id] = similarity
                    min_key = min(top_n, key=top_n.get)
                    top_n.pop(min_key)

            for trainer_id in top_n:
                total_similarity += int(top_n[trainer_id])

            big_term = 0
            for trainer_id in top_n:
                big_term += (int(top_n[trainer_id]) * ( train_list[int(trainer_id)-1].ratings[target] - train_list[int(trainer_id)-1].average_rating ))

            if total_similarity != 0:
                big_term = big_term/abs(total_similarity)
            else:
                big_term = 0

            processed_user.ratings[target] = int(round(test_user.average_rating + big_term))


    return processed_user

def apply_IUF(train_list_input, inv_ratings_input):
    train_list = train_list_input
    inv_ratings = inv_ratings_input
    for movie_id in inv_ratings:
        m_j = len(inv_ratings[movie_id]) #number of users that have rated item j
        IUF = math.log(200/m_j)
        for user_id in inv_ratings[movie_id]:
            train_list[int(user_id)-1].add_rating(movie_id, train_list[int(user_id)-1].get_rating(movie_id) * IUF)


    set_characteristics(train_list)
    return train_list       
            
        
        
def calculate_pearson_prediction_IUF(test_user, train_list, IUF_train_list, inv_ratings):
    processed_user = test_user
    
    for target in test_user.targets:
        if target in inv_ratings:
            n = 75
            top_n = {}
            total_similarity = 0
            min_key = str()
            for trainer_id in inv_ratings[target]:
                if (len(inv_ratings[target]) == 0):
                    print("There is no rating data for movie ", target)

                
                similarity = calculate_pearson_correlation(test_user, IUF_train_list[int(trainer_id)-1])
                if train_list[int(trainer_id)-1].ratings[target] > 5:
                    print("User ID: {}, Rating: {}, Similarity: {}".format(trainer_id, train_list[int(trainer_id)-1].ratings[target], similarity))
                if len(top_n) < n:
                    top_n[trainer_id] = similarity
                else:
                    top_n[trainer_id] = similarity
                    min_key = min(top_n, key=top_n.get)
                    top_n.pop(min_key)

            for trainer_id in top_n:
                total_similarity += int(top_n[trainer_id])

            big_term = 0
            for trainer_id in top_n:
                big_term += (int(top_n[trainer_id]) * ( train_list[int(trainer_id)-1].ratings[target] - train_list[int(trainer_id)-1].average_rating ))

            if total_similarity != 0:
                big_term = big_term/ total_similarity
            else:
                big_term = 0

            processed_user.ratings[target] = int(round(test_user.average_rating + big_term))


    return processed_user

def calculate_pearson_prediction_CA(test_user, train_list, inv_ratings):
    processed_user = test_user
    count = 0
    
    for target in test_user.targets:
        if target in inv_ratings:
            n = 75
            top_n = {}
            total_similarity = 0
            min_key = str()
            for trainer_id in inv_ratings[target]:
                if (len(inv_ratings[target]) == 0):
                    print("There is no rating data for movie ", target)
                    
                similarity = calculate_pearson_correlation(test_user, train_list[int(trainer_id)-1])
                if train_list[int(trainer_id)-1].ratings[target] > 5:
                    print("User ID: {}, Rating: {}, Similarity: {}".format(trainer_id, train_list[int(trainer_id)-1].ratings[target], similarity))
                if len(top_n) < n:
                    top_n[trainer_id] = similarity
                else:
                    top_n[trainer_id] = similarity
                    min_key = min(top_n, key=top_n.get)
                    top_n.pop(min_key)

            for trainer_id in top_n:
                top_n[trainer_id] = abs(int(top_n[trainer_id]))**(2.5-1)# CASE AMPLIFICATION STEP
                total_similarity += abs(int(top_n[trainer_id]))

            big_term = 0
            for trainer_id in top_n:
                big_term += (int(top_n[trainer_id]) * ( train_list[int(trainer_id)-1].ratings[target] - train_list[int(trainer_id)-1].average_rating ))

            if total_similarity != 0:
                big_term = big_term/ abs(total_similarity)
            else:
                big_term = 0

            processed_user.ratings[target] = int(round(test_user.average_rating + big_term))


    return processed_user


def calculate_inverted_list(train_list):
    inverted_list = {}

    for trainer in train_list:
        for key in trainer.ratings:
            if trainer.ratings[key] != 0:
                if key in inverted_list:
                    inverted_list[key].append(trainer.id)
                else:
                    inverted_list[key] = [trainer.id]

    with open("inv_ratings.txt", "w") as file_name:
        for movie_id in inverted_list:
            file_name.write("Movie ID: {}\n".format(movie_id))
            for user_id in inverted_list[movie_id]:
                if train_list[int(user_id)-1].get_rating(movie_id) > 5:
                    file_name.write("User ID: {}. Rating: {} \n".format(user_id, train_list[int(user_id)-1].get_rating(movie_id)))

##    for movie_id in inverted_list:
##        common_users = inverted_list[movie_id]
##        for user_id in inverted_list[movie_id]:
##            check_common(user_id, train_list, common_users, inverted_list)
            
    
        
    for movie_id in inverted_list:
        ratings = {}
        for user_id in inverted_list[movie_id]:
            ratings[user_id] = train_list[int(user_id)-1].get_rating(movie_id)

        movie_list.append(movie(movie_id, ratings, ratings, 1))
               
    for movie_id in movie_list:
        movie_id.average_rating = movie_id.calc_average_rating()
        movie_id.length = movie_id.calc_euclidean_length()
        
    return inverted_list

def calculate_inverted_list_movie():
    inverted_list = {}

    for movie_id in movie_list:
        for key in movie_id.ratings:
            if movie_id.ratings[key] != 0:
                if key in inverted_list:
                    inverted_list[key].append(movie_id.id)
                else:
                    inverted_list[key] = [movie_id.id]


    return inverted_list
    


##def calculate_adjusted_cosine_similarity(current_movie, other_movie):
##    test_list = []
##    train_list = []
##    for movie_id in test_user.ratings:
##        if movie_id in train_user.ratings:
##            if int(test_user.ratings[movie_id]) != 0 and int(train_user.ratings[movie_id]) != 0:
##                test_list.append(int(test_user.ratings[movie_id]))
##                train_list.append(int(train_user.ratings[movie_id]))
##
##    if len(test_list) >= 2:
##        result =  1 - spatial.distance.cosine(test_list, train_list)
##    else:
##        result = 0
##
##    if math.isnan(result):
##        return 0
##    else:
##        return result

def calculate_item_cosine(test_movie_input, train_list_input, inv_ratings_input):
    train_list = train_list_input
    test_movie = test_movie_input
    processed_movie = test_movie
    inv_ratings = inv_ratings_input       
    
    for target in test_movie.targets:

        if target in inv_ratings_movie:
            n = 5
            top_n = {}
            total_similarity = 0
            predict_rating = 0
            min_key = str()
            #Select and get top k similar movies
            for movie_id in inv_ratings[target]:
                if (len(inv_ratings[target]) == 0):
                    print("There is no rating data for movie ", target)

                                    
                similarity = calculate_cosine_similarity(test_movie, movie_list[int(movie_id)-1])
                if len(top_n) < n:
                    top_n[movie_id] = similarity
                else:
                    top_n[movie_id] = similarity
                    min_key = min(top_n, key=top_n.get)
                    top_n.pop(min_key)

            #compute summation of w (denominator):
            for movie_id in top_n:
                total_similarity += top_n[movie_id]

            #compute summation of product of u and r (numerator):
            for movie_id in top_n:
                if total_similarity != 0:
                    predict_rating += (top_n[movie_id]*train_list[int(movie_id)-1].ratings[target])

            if total_similarity != 0:
                predict_rating = predict_rating/total_similarity
            elif total_similarity == 0: # ASSUME THAT WITHOUT OTHER DATA, THE USER GIVES THE AVG RATING
                predict_rating = test_movie.average_rating
                print(test_user.average_rating)

            
            processed_movie.ratings[target] = round(predict_rating)

                        
    return processed_movie

def calculate_personal(test_user, train_list, IUF_train_list, inv_ratings):
    processed_user = test_user
    
    for target in test_user.targets:
        if target in inv_ratings:
            n = 90
            top_n = {}
            total_similarity = 0
            min_similarity = 1
            min_key = str()
            for trainer_id in inv_ratings[target]:
                if (len(inv_ratings[target]) == 0):
                    print("There is no rating data for movie ", target)

                if train_list[int(trainer_id)-1].ratings[target] > 5:
                    print("User ID: {}, Rating: {}, Similarity: {}".format(trainer_id, train_list[int(trainer_id)-1].ratings[target], similarity))
                similarity = calculate_cosine_similarity(test_user, IUF_train_list[int(trainer_id)-1])
                
                if len(top_n) < n:
                    top_n[trainer_id] = similarity
                else:
                    top_n[trainer_id] = similarity
                    min_key = min(top_n, key=top_n.get)
                    top_n.pop(min_key)

            for trainer_id in top_n:
                top_n[trainer_id] = abs(int(top_n[trainer_id]))**(2.5-1)# CASE AMPLIFICATION STEP
                total_similarity += abs(int(top_n[trainer_id]))
                

            big_term = 0
            for trainer_id in top_n:
                big_term += (int(top_n[trainer_id]) * ( train_list[int(trainer_id)-1].ratings[target] - train_list[int(trainer_id)-1].average_rating ))

            if total_similarity != 0:
                big_term = big_term/abs(total_similarity)
            else:
                big_term = 0

            processed_user.ratings[target] = round(test_user.average_rating + big_term)


    return processed_user



def parser(file_path, train_test):
    if train_test == 0:
        trainers = []

        with open(file_path, "r") as train_file:
            i = 1
            while True:
                line = train_file.readline()
                if not line:
                    break
                line = line.split()
                user_ratings = line[0:1000]
                trainers.append(user(i, user_ratings, user_ratings, 0))
                i+=1
        #for trainer_id in trainers:
         #   for target in trainer_id.ratings:
          #      if trainers[int(trainer_id.id)-1].ratings[target] > 5:
                    #print("User ID: {}, Rating: {}".format(trainer_id.id, trainers[int(trainer_id.id)-1].ratings[target]))
        return trainers
    
    elif train_test == 1:
        testers = []
        users = {}

        with open(file_path, "r") as test_file:
                        
            for line in test_file:
                tokens = line.split()
                user_id = tokens[0]
                movie_id = tokens[1]
                rating = tokens[2]

                if user_id not in users:
                    users[user_id] = user(user_id, {movie_id:rating}, {movie_id:rating}, 1)
                else:
                    users[user_id].add_rating(movie_id, int(rating))
            
        for key in users:
            testers.append(users[key])

        with open("ex.txt", "w") as file_name:
            for tester_obj in testers:
                for movie_id in tester_obj.ratings:
                    #if movie_id in tester_obj.targets:
                        output_ratings = tester_obj.ratings[movie_id]
                        file_name.write("{} {} {}\n".format(tester_obj.id, movie_id, output_ratings))

        return testers



        
def processing(list_testing_users, list_training_users, inv_ratings, IUF_train_list, task):
    count = 1
    results = []
    file_name = ""
    
    
    if task == 0: #COSINE SIMILARITY PROCESSING

##        for trainer in list_training_users:
##            for target in trainer.ratings:
##                if list_training_users[int(trainer.id)-1].ratings[target] > 5:
##                    print("User ID: {}, Rating: {}".format(trainer.id, list_training_users[int(trainer.id)-1].ratings[target]))

        if int(list_testing_users[0].id) > 400:
            file_name = "resultcosine20"
        elif int(list_testing_users[0].id) > 300:
            file_name = "resultcosine10"
        else:
            file_name = "resultcosine5"
        for tester in list_testing_users:
            if (count % 10) == 0:
                print("Completed {}/{} in {}.".format(count, len(list_testing_users), file_name))
            results.append(calculate_cosine_prediction(tester, list_training_users, inv_ratings))
            count += 1
        write_file(results, file_name+".txt")
        print("File written")

        
    elif task == 1: #PEARSON CORRELATION PROCESSING
        if int(list_testing_users[0].id) > 400:
            file_name = "resultpearson20"
        elif int(list_testing_users[0].id) > 300:
            file_name = "resultpearson10"
        else:
            file_name = "resultpearson5"
        for tester in list_testing_users:
            if (count % 10) == 0:
                print("Completed {}/{} in {}.".format(count, len(list_testing_users), file_name))
            results.append(calculate_pearson_prediction(tester, list_training_users, inv_ratings))
            count += 1
        write_file(results, file_name+".txt")
        print("File written")

    elif task == 2: #PEARSON CORRELATION WITH IUF PROCESSING
        if int(list_testing_users[0].id) > 400:
            file_name = "resultpearson_iuf20"
        elif int(list_testing_users[0].id) > 300:
            file_name = "resultpearson_iuf10"
        else:
            file_name = "resultpearson_iuf5"
        for tester in list_testing_users:
            if (count % 10) == 0:
                print("Completed {}/{} in {}.".format(count, len(list_testing_users), file_name))
            results.append(calculate_pearson_prediction_IUF(tester, list_training_users, IUF_train_list, inv_ratings))
            count += 1
        write_file(results, file_name+".txt")
        print("File written")

    elif task == 3: #PEARSON CORRELATION WITH CASE AMPLIFICATION
        if int(list_testing_users[0].id) > 400:
            file_name = "resultpearson_ca20"
        elif int(list_testing_users[0].id) > 300:
            file_name = "resultpearson_ca10"
        else:
            file_name = "resultpearson_ca5"
        for tester in list_testing_users:
            if (count % 10) == 0:
                print("Completed {}/{} in {}.".format(count, len(list_testing_users), file_name))
            results.append(calculate_pearson_prediction_CA(tester, list_training_users, inv_ratings))
            count += 1
        write_file(results, file_name+".txt")
        print("File written")

    elif task == 4: #ITEM-BASE COSINE SIMILARITY
        if int(list_testing_users[0].id) > 400:
            file_name = "resultitem_cosine_20"
        elif int(list_testing_users[0].id) > 300:
            file_name = "resultitem_cosine_10"
        else:
            file_name = "resultitem_cosine_5"
        for test_movie in movie_list:
            if (count % 10) == 0:
                print("Completed {}/{} in {}.".format(count, len(list_testing_users), file_name))
            results.append(calculate_item_cosine(test_movie, list_training_users, inv_ratings))
            count += 1
        write_file_item(results, list_testing_users,  file_name+".txt")
        print("File written")

    elif task == 5: #PERSONAL ALGORITHM: CASE AMPLIFICATION, PEARSON CORRELATION
        if int(list_testing_users[0].id) > 400:
            file_name = "resultpersonal_20"
        elif int(list_testing_users[0].id) > 300:
            file_name = "resultpersonal_10"
        else:
            file_name = "resultpersonal_5"
        for tester in list_testing_users:
            if (count % 10) == 0:
                print("Completed {}/{} in {}.".format(count, len(list_testing_users), file_name))
            results.append(calculate_personal(tester, list_training_users, IUF_train_list, inv_ratings))
            count += 1
        write_file(results, file_name+".txt")
        print("File written")

        
def write_file(test_list, output_path):
    with open(output_path, "w") as file_name:
        for tester_obj in test_list:
            for movie_id in tester_obj.ratings:
                if movie_id in tester_obj.targets:
                    output_ratings = tester_obj.ratings[movie_id]
                    if int(output_ratings) < 1:
                       output_ratings = 1
                    if int(output_ratings) > 5:
                       output_ratings = 5
                    file_name.write("{} {} {}\n".format(tester_obj.id, movie_id, output_ratings))

def write_file_item(movie_results, test_list, output_path):

    for user_id in test_list:
        for movie_id in user_id.targets:
            for movie_obj in movie_results:
                if movie_id == movie_obj.id and user_id in movie_obj.ratings:
                    user_id.ratings[movie_id] = movie_obj.average_rating

                
    with open(output_path, "w") as file_name:
        for tester_obj in test_list:
            for movie_id in tester_obj.ratings:
                #if movie_id in tester_obj.targets:
                    output_ratings = tester_obj.ratings[movie_id]
                    if int(output_ratings) < 1:
                       output_ratings = 1
                    if int(output_ratings) > 5:
                       output_ratings = 5
                    file_name.write("{} {} {}\n".format(tester_obj.id, movie_id, output_ratings))


def main():
    train_file = "train.txt"
    testing_files = ["test5.txt", "test10.txt", "test20.txt"]
    list_training_users = parser(train_file, 0)
    IUF_train_list = parser(train_file, 0)
    list_testing_users_5 = parser(testing_files[0], 1)
    list_testing_users_10 = parser(testing_files[1], 1)
    list_testing_users_20 = parser(testing_files[2], 1)

    list_training_users = set_characteristics(list_training_users)
    list_testing_users_5 = set_characteristics(list_testing_users_5)
    list_testing_users_10 = set_characteristics(list_testing_users_10)
    list_testing_users_20 = set_characteristics(list_testing_users_20)
 
    inv_ratings = calculate_inverted_list(list_training_users)
    IUF_train_list = apply_IUF(IUF_train_list, inv_ratings)
    inv_ratings_movie = calculate_inverted_list_movie()

    #TASK = 0: cosine similarity
    processing(list_testing_users_5, list_training_users, inv_ratings, IUF_train_list, 0)
    processing(list_testing_users_10, list_training_users, inv_ratings, IUF_train_list, 0)
    processing(list_testing_users_20, list_training_users, inv_ratings, IUF_train_list, 0)

    #TASK = 1: pearson correlation
    processing(list_testing_users_5, list_training_users, inv_ratings, IUF_train_list, 1)
    processing(list_testing_users_10, list_training_users, inv_ratings, IUF_train_list, 1)
    processing(list_testing_users_20, list_training_users, inv_ratings, IUF_train_list, 1)

    #TASK = 2: pearson correlation (Inverse User Frequency)
    processing(list_testing_users_5, list_training_users, inv_ratings, IUF_train_list, 2)
    processing(list_testing_users_10, list_training_users, inv_ratings, IUF_train_list, 2)
    processing(list_testing_users_20, list_training_users, inv_ratings, IUF_train_list, 2)

    #TASK = 3: pearson correlation (Case Amplification)
    processing(list_testing_users_5, list_training_users, inv_ratings, IUF_train_list, 3)
    processing(list_testing_users_10, list_training_users, inv_ratings, IUF_train_list, 3)
    processing(list_testing_users_20, list_training_users, inv_ratings, IUF_train_list, 3)

    #TASK = 4: item-based cosine
    processing(list_testing_users_5, list_training_users, inv_ratings, IUF_train_list, 4)
    processing(list_testing_users_10, list_training_users, inv_ratings, IUF_train_list, 4)
    processing(list_testing_users_20, list_training_users, inv_ratings, IUF_train_list, 4)

    #TASK = 5: personal
    processing(list_testing_users_5, list_training_users, inv_ratings, IUF_train_list, 5)
    processing(list_testing_users_10, list_training_users, inv_ratings, IUF_train_list, 5)
    processing(list_testing_users_20, list_training_users, inv_ratings, IUF_train_list, 5)
    
    sys.exit(0)

main()
