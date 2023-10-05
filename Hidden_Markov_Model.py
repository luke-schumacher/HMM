import os
import pandas as pd
import numpy as np
from hmmlearn.hmm import MultinomialHMM, CategoricalHMM
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load_data_instances():
    print(os.getcwd())
    data_dir = "./data"

    dfs = []
    file_names = os.listdir(data_dir)

    print("Loading instances:")
    print(file_names)

    for name in file_names:
        temp_path = os.path.join(data_dir, name)
        dfs.append(pd.read_csv(temp_path, sep=";"))

    return dfs


def insert_start_row(df):

    df["start"] = False

    start_row = df.iloc[:1].copy()

    for col in start_row.columns:
        col_type = str(start_row.dtypes[col])
        if col_type == "bool":
            start_row[col] = False
        elif col_type == "object":
            if col != "part":
                start_row[col] = "False"
        elif "int" in col_type:
            start_row[col] = 0

    start_row["start"] = True
    df = pd.concat([start_row, df], ignore_index=True)

    return df


def insert_end_row(df):
    df["end"] = False

    end_row = df.iloc[-1:].copy()

    for col in end_row.columns:
        col_type = str(end_row.dtypes[col])
        if col_type == "bool":
            end_row[col] = False
        elif col_type == "object":
            if col != "part":
                end_row[col] = "False"
        elif "int" in col_type:
            end_row[col] = 0

    end_row["end"] = True
    df = pd.concat([df, end_row], ignore_index=True)

    return df


# Preprocessing
def preprocess_data(dataframe, categories_dict):
    drop_columns = ['serial', 'start_date', 'end_date']
    dataframe = dataframe.drop(columns=drop_columns)

    dataframe = insert_start_row(dataframe)
    dataframe = insert_end_row(dataframe)

    # Strip leading/trailing whitespaces from column names
    dataframe.columns = dataframe.columns.str.strip()

    condition_cols = ["part", "pseudo", "start", "end", "sequence", "aborted", "seconds"]
    element_cols = [col for col in dataframe.columns if "element" in col]
    item_cols = [col for col in dataframe.columns if "item" in col]

    no_process_cols = ["part"]
    numerical_cols = ["seconds"]
    bool_columns = element_cols + item_cols + ["pseudo", "aborted", "start", "end"]
    cat_columns = [col for col in condition_cols if col not in (bool_columns + numerical_cols + no_process_cols)]

    df_no_process = dataframe[no_process_cols]
    df_numerical = dataframe[numerical_cols].astype(int)
    df_bool = dataframe[bool_columns].astype(int)
    df_categorical = dataframe[cat_columns].copy()

    for col in cat_columns:
        df_categorical[col] = pd.Categorical(df_categorical[col].values,
                                             categories=["False"] + categories_dict[col])

    # Convert categorical values to integers
    df_categorical = df_categorical.apply(lambda x: x.cat.codes)

    # # Adjust categorical values to be nonnegative integers
    # df_categorical += 1

    # print("numerical df")
    # print(df_numerical.head())
    #
    # print("bool df")
    # print(df_bool.head())
    #
    # print("categorical df")
    # print(df_categorical.head())

    # Combine the numeric and categorical dataframes
    df_processed = pd.concat([df_no_process, df_categorical, df_numerical, df_bool], axis=1)

    df_processed = df_processed[condition_cols + element_cols + item_cols]

    return df_processed


def add_last_rows(df, new_length):
    assert len(df) < new_length

    remaining_df = df.iloc[-1:].copy()
    remaining_df = [remaining_df for _ in range(new_length - len(df))]
    remaining_df = pd.concat(remaining_df, ignore_index=True)

    new_df = pd.concat([df, remaining_df], ignore_index=True)
    # print(new_df.head())

    return new_df


def generate_training_data(dfs, cat_cols=("sequence",), repeat_end=2):
    training_data = []

    cat_values = dict(zip(cat_cols, [[] for _ in cat_cols]))
    lengths = []

    for df in dfs:
        lengths.append(len(df))
        for col in cat_cols:
            cat_values[col].extend(list(df[col].unique()))

    max_length = max(lengths) + repeat_end

    for col in cat_cols:
        cat_values[col] = list(sorted(set(cat_values[col])))

    for df in dfs:
        # Preprocess the data
        df_processed = preprocess_data(df, cat_values)

        # Extract the sequence of symbols from the preprocessed data
        condition_columns = ["start", "end", "sequence"]
        binary_columns = [col for col in df_processed.columns if "element" in col]

        df_for_model = df_processed[condition_columns + binary_columns]
        if len(df_for_model) < max_length:
            df_for_model = add_last_rows(df_for_model, max_length)

        training_data.append(df_for_model)

    # return np.concatenate(training_data), [item.shape[0] for item in training_data]
    return training_data, [item.shape[0] for item in training_data]


def test():
    # X1 = [[0.5], [1.0], [-1.0], [0.42], [0.24]]
    # X2 = [[2.4], [4.2], [0.5], [-0.24]]
    #
    # X = np.concatenate([X1, X2])
    # lengths = [len(X1), len(X2)]
    #
    # print(X.shape)
    # print(lengths)
    #
    # # Train the HMM model
    # model = CategoricalHMM(n_components=2,
    #                        # n_iter=100,
    #                        # tol=0.01,
    #                        # algorithm='viterbi',
    #                        # n_trials=train_data.shape[0]
    #                        )
    #
    # model.fit(X, lengths)

    from hmmlearn import hmm

    # For this example, we will model the stages of a conversation,
    # where each sentence is "generated" with an underlying topic, "cat" or "dog"
    states = ["cat", "dog"]
    id2topic = dict(zip(range(len(states)), states))
    # we are more likely to talk about cats first
    start_probs = np.array([0.6, 0.4])

    # For each topic, the probability of saying certain words can be modeled by
    # a distribution over vocabulary associated with the categories

    vocabulary = ["tail", "fetch", "mouse", "food"]
    # if the topic is "cat", we are more likely to talk about "mouse"
    # if the topic is "dog", we are more likely to talk about "fetch"
    emission_probs = np.array([[0.25, 0.1, 0.4, 0.25],
                               [0.2, 0.5, 0.1, 0.2]])

    # Also assume it's more likely to stay in a state than transition to the other
    trans_mat = np.array([[0.8, 0.2], [0.2, 0.8]])

    # Pretend that every sentence we speak only has a total of 5 words,
    # i.e. we independently utter a word from the vocabulary 5 times per sentence
    # we observe the following bag of words (BoW) for 8 sentences:
    observations = [["tail", "mouse", "mouse", "food", "mouse"],
                    ["food", "mouse", "mouse", "food", "mouse"],
                    ["tail", "mouse", "mouse", "tail", "mouse"],
                    ["food", "mouse", "food", "food", "tail"],
                    ["tail", "fetch", "mouse", "food", "tail"],
                    ["tail", "fetch", "fetch", "food", "fetch"],
                    ["fetch", "fetch", "fetch", "food", "tail"],
                    ["food", "mouse", "food", "food", "tail"],
                    ["tail", "mouse", "mouse", "tail", "mouse"],
                    ["fetch", "fetch", "fetch", "fetch", "fetch"]]

    # Convert "sentences" to numbers:
    vocab2id = dict(zip(vocabulary, range(len(vocabulary))))

    def sentence2counts(sentence):
        ans = []
        for word, idx in vocab2id.items():
            count = sentence.count(word)
            ans.append(count)
        return ans

    X = []
    for sentence in observations:
        row = sentence2counts(sentence)
        X.append(row)

    data = np.array(X, dtype=int)

    # pretend this is repeated, so we have more data to learn from:
    lengths = [len(X)] * 5
    sequences = np.tile(data, (5, 1))

    print(sequences.shape)
    print(sequences)

    # Set up model:
    model = hmm.MultinomialHMM(n_components=len(states),
                               n_trials=len(observations[0]),
                               n_iter=50,
                               init_params='')

    model.n_features = len(vocabulary)
    model.startprob_ = start_probs
    model.transmat_ = trans_mat
    model.emissionprob_ = emission_probs
    model.fit(sequences, lengths)
    logprob, received = model.decode(sequences)

    print("Topics discussed:")
    print([id2topic[x] for x in received])

    print("Learned emission probs:")
    print(model.emissionprob_)

    print("Learned transition matrix:")
    print(model.transmat_)

    # Try to reset and refit:
    new_model = hmm.MultinomialHMM(n_components=len(states),
                                   n_trials=len(observations[0]),
                                   n_iter=50, init_params='ste')

    new_model.fit(sequences, lengths)
    logprob, received = new_model.decode(sequences)

    print("\nNew Model")
    print("Topics discussed:")
    print([id2topic[x] for x in received])

    print("Learned emission probs:")
    print(new_model.emissionprob_)

    print("Learned transition matrix:")
    print(new_model.transmat_)


def transform_to_one_hot(df_list):
    output_arrays = []
    total_series = None

    temp_dfs = []

    for df in df_list:
        temp_df = df.copy()

        # temp_df = temp_df.astype(str).sum(axis=1)
        temp_df = temp_df.astype(str).apply(lambda x: "_".join(list(x)), axis=1)

        temp_dfs.append(temp_df.copy())

        if total_series is None:
            total_series = temp_df.copy()
        else:
            total_series = pd.concat([total_series, temp_df], ignore_index=True)
    #
    # print(total_series.head(30))

    first_value = total_series.iloc[0]
    uniques = list(sorted(set(total_series.unique())))
    uniques = [first_value] + list(np.setdiff1d(uniques, [first_value]))
    n_uniques = len(uniques)

    one_hot_translations = [dict(zip(uniques, range(n_uniques))),
                            dict(zip(range(n_uniques), uniques))]

    for df in temp_dfs:
        temp_df = df.apply(lambda x: one_hot_translations[0][x])

        temp_array = np.zeros((len(temp_df), n_uniques))

        for i in range(len(temp_df)):
            temp_array[i, temp_df.iloc[i]] = 1

        output_arrays.append(temp_array.astype(int))

    return output_arrays, one_hot_translations


def re_transform_from_one_hot_encoding(array, translations):
    position_array = np.argmax(array, axis=1)
    strings = pd.Series(position_array).apply(lambda x: translations[1][x])

    array = []
    for sub_string in strings:
        temp_array = np.array(sub_string.split("_")).astype(int)
        array.append(temp_array)

    array = np.array(array)
    #
    # print(array)

    return array


def post_process_sample(sample_sequence, column_names):
    sample_df = pd.DataFrame(sample_sequence, columns=column_names)

    if 1 in sample_df.end.unique():
        max_length = sample_df[sample_df.end == 1].index[0] + 1

        return sample_df.iloc[:max_length]

    return sample_df


def sample(model, sample_length, translations, columns):
    # Generate a sample sequence using the trained model
    sample_sequence, sample_hidden_states_sequence = model.sample(n_samples=sample_length, currstate=0)

    # for col in sample_sequence:
    #     print(col)
    #
    # print(sample_hidden_states_sequence)

    transformed_sample = re_transform_from_one_hot_encoding(sample_sequence, translations)
    adjusted_sample_df = post_process_sample(transformed_sample, columns)

    return adjusted_sample_df, sample_hidden_states_sequence


def main(n_components=None,
         n_repeat_end_line=2,
         n_iter=100,
         tol=0.01):

    # load data instances
    print("Loading data instances")
    dfs = load_data_instances()

    # generate training data from instances
    print("Generating training data from instances")
    train_data, train_lengths = generate_training_data(dfs, repeat_end=n_repeat_end_line)
    columns = list(train_data[0].columns)

    print("Using columns:")
    print(columns)

    # transform training data to multinomial notation for the product of all columns
    print("Transforming training data for Multinomial HMM")
    multinomial_train_data, translations = transform_to_one_hot(train_data)
    multinomial_train_data = np.concatenate(multinomial_train_data)

    print(f"Found {len(translations[0])} unique emission values for the product of {len(columns)} variables")

    # print(multinomial_train_data.shape)
    # print(train_lengths)
    # print(multinomial_train_data[:10])

    # Train the HMM model
    if n_components is None:
        n_components_to_use = len(translations[0])
        init_params = "t"
    else:
        n_components_to_use = n_components
        init_params = "te"

    print(f"Using HMM with {n_components_to_use} hidden components")

    print("Building HMM model")
    model = MultinomialHMM(n_components=n_components_to_use,
                           n_iter=n_iter,
                           tol=tol,
                           algorithm='viterbi',
                           n_trials=1,
                           init_params=init_params,
                           )

    start_probs = np.zeros(n_components_to_use)
    start_probs[0] = 1.0

    model.startprob_ = start_probs

    if n_components is None:
        # emission_probs = np.ones((len(translations[0]), len(translations[0]))) / len(translations[0])

        emission_probs = np.zeros((len(translations[0]), len(translations[0])))
        for i in range(len(translations[0])):
            emission_probs[i, i] = 1.0

        model.emissionprob_ = emission_probs

    # trans_mat = np.ones((n_components_to_use, n_components_to_use)) / n_components_to_use
    # trans_mat[:, 0] = 0.0
    # trans_mat[-1, :] = 0.0
    # trans_mat[-1, -1] = 1.0
    #
    # model.transmat_ = trans_mat

    print("Training / Fitting HMM model on provided training data")
    model.fit(multinomial_train_data, train_lengths)

    # evaluation
    print("Evaluation of trained HMM model")

    print("Results for train instances")
    sub_index = 0
    for i in range(10):
        top_index = sub_index + train_lengths[i]

        sequence = multinomial_train_data[sub_index:top_index]
        transformed_sequence = re_transform_from_one_hot_encoding(sequence, translations)
        adjusted_sequence_df = post_process_sample(transformed_sequence, columns)

        logprob, received = model.decode(sequence)

        sub_index = top_index

        print(f"Hidden states for train instance {i}")
        print([x for x in received])
        print(adjusted_sequence_df)

    # print("Learned emission probs:")
    # print(model.emissionprob_)
    #
    # print("Learned transition matrix:")
    # print(model.transmat_)

    # # Calculate the total number of observations in the dataset
    # total_observations = len(df)
    #

    print("Generating samples with trained HMM model")
    for i in range(10):
        print(f"sample {i}")
        sample_df, sample_hidden_states = sample(model, train_lengths[0], translations, columns)

        print(sample_hidden_states)
        print(sample_df)

    #
    # # Save the sample sequence as an image
    # plt.figure(figsize=(10, 5))
    # plt.imshow(sample_sequence.T, cmap='gray', aspect='auto')
    # plt.xlabel('Time Step')
    # plt.ylabel('Hidden State')
    # plt.title('Generated Sequence')
    # plt.savefig('generated_sequence.png')


if __name__ == '__main__':

    # test()
    n_components = None
    main(n_components=n_components,
         n_repeat_end_line=10,
         n_iter=100,
         tol=0.01
         )
    
    
