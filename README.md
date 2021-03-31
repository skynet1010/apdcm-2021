# Results of our NAS approach
1. cd into the root folder of this repository.
2. create a python environment by the following command:

`conda env create -f code/environment.yaml`

3. activate environment:

`conda activate apdcm`

4. cd in code folder:

`cd code`

5. execute the following bash script:

`bash bash execute_models.sh`

Output should look similar as follows:
```
CREATE model...
CREATE layer: layer_0
CREATE layer: layer_1
CREATE layer: layer_2
CREATE layer: layer_3
CREATE layer: layer_4
CREATE layer: layer_5
CREATE layer: layer_6
MODEL created!
Ship model to device...
MODEL successfully shipped to device!
#Test 1/20
#Test 2/20
.
.
.
#Test 20/20
MNIST:
Free parameters: 966215
Average accuracy: 99.75%
```
The Models are stored in the folder models.
