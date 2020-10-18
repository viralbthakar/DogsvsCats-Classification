# DogsvsCats-Classification

## Download and Prepare Dataset

    For this video, we are going to use the Dog vs Cat Image classification dataset available on Kaggle. You can directly download the zip file using the link from the description or setup the Kaggle API on your system and use the following command to download the zip file. 

    ```jsx
    mkdir ~/DogsvsCats
    cd ~/DogsvsCats
    mkdir data
    cd data
    kaggle competitions download -c dogs-vs-cats
    unzip dogs-vs-cats.zip
    unzip train.zip
    unzip test1.zip
    rm dogs-vs-cats.zip train.zip test1.zip
    cd ..
    tree -d
    ls data/train/ -U | head -20
    identify ./data/train/cat.0.jpg
    ```

    Our project structure looks something like this 

    ![imgs/Screenshot_from_2020-10-17_15-27-22.png](imgs/Screenshot_from_2020-10-17_15-27-22.png)

    And the images are placed under the `train` directory and they look something like this

    ![imgs/Screenshot_from_2020-10-17_15-31-01.png](imgs/Screenshot_from_2020-10-17_15-31-01.png)

    We can look into the brief details about any image sample.

    ![imgs/Screenshot_from_2020-10-17_15-35-18.png](imgs/Screenshot_from_2020-10-17_15-35-18.png)

