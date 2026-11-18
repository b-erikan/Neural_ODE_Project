from sklearn.model_selection import train_test_split



def split_trajectory_ids(trajectory_ids, train_ratio, val_ratio, seed):
    '''This function is created to randomly split main trajectory IDs into subsections. 
    We will then pick all the corresponding sub-trajectories from our processed dataset to ensure consistency.
    First extract the train IDs, then split the remainder into validation & test.
    I implemented this as a seperate function in case I want to change the split strategy later on.'''  
    train_ids, temp_ids = train_test_split(
        trajectory_ids,
        test_size=1.0 - train_ratio,
        random_state=seed,
        shuffle=True
    )
    val_fraction = val_ratio / (1.0 - train_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=1.0 - val_fraction,
        random_state=seed,
        shuffle=True
    )

    return train_ids, val_ids, test_ids 