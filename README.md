# t5_11
housing our model example of fine tuning an 11B t5 with FSDP to create a world-class grammar checker.
<img width="663" alt="children_correction" src="https://user-images.githubusercontent.com/46302957/165164090-5421958f-8ac5-49dc-b8f1-50cd6f26047a.png">


### to get going...
~~~ 
pip install -r requirements.txt
~~~

a large and small dataset are already present in the project (grammar_train.csv = small, gtrain_150K.csv = large). 

### to baseline your environment or this model (adjust nproc to equal your gpu count):
~~~
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint="localhost:5679" main_benchmark.py  
~~~

On an A100 (p4d.24xlarge) you should expect to see:

<img width="311" alt="benchmark_t5" src="https://user-images.githubusercontent.com/46302957/169574633-e3563cf7-c30c-4bf1-8a98-7c99a621f228.png">


To train with mp spawn:
~~~
python main.py
~~~

Or better, with torchrun:
~~~
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint="localhost:5679" main_elastic.py  
~~~

You can control the model size, dataset size, batch size, etc. all in the config/defaults.py
