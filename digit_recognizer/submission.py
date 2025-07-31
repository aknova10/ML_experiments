import pandas as pd
from numpy.typing import NDArray

# prediction = model.predict(test_df)
def create_submission(prediction: NDArray, one_hot : bool = False) -> None:
    data = []
    i = 1
    if one_hot:
        for pred in prediction:
            data.append([i, pred.argmax()])
            i+=1
    else:
        for pred in prediction:
            data.append([i, pred])
            i+=1
    # np.where(prediction==1)
    # len(data)
    submission_df = pd.DataFrame(data, columns=['ImageId', 'Label'])
    submission_df.to_csv("submission.csv", index=False)