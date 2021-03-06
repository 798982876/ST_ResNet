# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import data.ST_ResNet.grpc_st.dist.st_resnet_pb2 as st__resnet__pb2


class StResnetStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Train = channel.unary_unary(
        '/GeoThinking.ML.StResnet/Train',
        request_serializer=st__resnet__pb2.execute_id.SerializeToString,
        response_deserializer=st__resnet__pb2.status.FromString,
        )
    self.Predict = channel.unary_unary(
        '/GeoThinking.ML.StResnet/Predict',
        request_serializer=st__resnet__pb2.execute_id.SerializeToString,
        response_deserializer=st__resnet__pb2.status.FromString,
        )


class StResnetServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Train(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Predict(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_StResnetServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Train': grpc.unary_unary_rpc_method_handler(
          servicer.Train,
          request_deserializer=st__resnet__pb2.execute_id.FromString,
          response_serializer=st__resnet__pb2.status.SerializeToString,
      ),
      'Predict': grpc.unary_unary_rpc_method_handler(
          servicer.Predict,
          request_deserializer=st__resnet__pb2.execute_id.FromString,
          response_serializer=st__resnet__pb2.status.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'GeoThinking.ML.StResnet', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
