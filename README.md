<h1>Classification with networks from HuggingFace Hub</h1>

<h2>Installation: </h2>

<p>Install Python 3.10.10</p>
<p>Install CUDA Toolkit 11.8</p>
<p>Install Pytorch 2.0</p>
<p>Install Other Requeriments: pip install -r requeriments.txt</p>

<h2>Organized Your Dataset: </h2>

```
|-- Dataset_Name/
|   |-- train/
|   |   |-- label_1/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- label_2/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- label_3/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- ...
|   |-- test/
|   |   |-- label_1/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- label_2/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- label_3/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- ...
|   |-- valid/
|   |   |-- label_1/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- label_2/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- label_3/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- ...
...
```

<h2>Execute: </h2>

<h3>Train: </h3>
<p>python trainer.py --dataset path_to_dataset --output_dir path_to_output_dir</p>

<h3>Test: </h3>
<p>python classify.py --image path_to_image --model path_to_checkpoint</p>
