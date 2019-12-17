import tensorflow as tf

sess = tf.Session()
with tf.gfile.FastGFile("pb_model/freeze_eval_graph.pb", 'rb') as f:
    # 使用tf.GraphDef()定义一个空的Graph
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    # Imports the graph from graph_def into the current default Graph.

tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name)
