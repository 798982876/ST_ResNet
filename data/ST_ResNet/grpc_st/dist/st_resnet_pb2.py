# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: st_resnet.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='st_resnet.proto',
  package='GeoThinking.ML',
  syntax='proto3',
  serialized_pb=_b('\n\x0fst_resnet.proto\x12\x0eGeoThinking.ML\"\x18\n\nexecute_id\x12\n\n\x02id\x18\x01 \x01(\t\"\x1d\n\x06status\x12\x13\n\x0bstatus_code\x18\x01 \x01(\t2\x8a\x01\n\x08StResnet\x12=\n\x05Train\x12\x1a.GeoThinking.ML.execute_id\x1a\x16.GeoThinking.ML.status\"\x00\x12?\n\x07Predict\x12\x1a.GeoThinking.ML.execute_id\x1a\x16.GeoThinking.ML.status\"\x00\x62\x06proto3')
)




_EXECUTE_ID = _descriptor.Descriptor(
  name='execute_id',
  full_name='GeoThinking.ML.execute_id',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='GeoThinking.ML.execute_id.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=35,
  serialized_end=59,
)


_STATUS = _descriptor.Descriptor(
  name='status',
  full_name='GeoThinking.ML.status',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status_code', full_name='GeoThinking.ML.status.status_code', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=61,
  serialized_end=90,
)

DESCRIPTOR.message_types_by_name['execute_id'] = _EXECUTE_ID
DESCRIPTOR.message_types_by_name['status'] = _STATUS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

execute_id = _reflection.GeneratedProtocolMessageType('execute_id', (_message.Message,), dict(
  DESCRIPTOR = _EXECUTE_ID,
  __module__ = 'st_resnet_pb2'
  # @@protoc_insertion_point(class_scope:GeoThinking.ML.execute_id)
  ))
_sym_db.RegisterMessage(execute_id)

status = _reflection.GeneratedProtocolMessageType('status', (_message.Message,), dict(
  DESCRIPTOR = _STATUS,
  __module__ = 'st_resnet_pb2'
  # @@protoc_insertion_point(class_scope:GeoThinking.ML.status)
  ))
_sym_db.RegisterMessage(status)



_STRESNET = _descriptor.ServiceDescriptor(
  name='StResnet',
  full_name='GeoThinking.ML.StResnet',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=93,
  serialized_end=231,
  methods=[
  _descriptor.MethodDescriptor(
    name='Train',
    full_name='GeoThinking.ML.StResnet.Train',
    index=0,
    containing_service=None,
    input_type=_EXECUTE_ID,
    output_type=_STATUS,
    options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Predict',
    full_name='GeoThinking.ML.StResnet.Predict',
    index=1,
    containing_service=None,
    input_type=_EXECUTE_ID,
    output_type=_STATUS,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_STRESNET)

DESCRIPTOR.services_by_name['StResnet'] = _STRESNET

# @@protoc_insertion_point(module_scope)
