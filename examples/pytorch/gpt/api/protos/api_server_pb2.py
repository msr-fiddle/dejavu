# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/api_server.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17protos/api_server.proto\x12\ndejavu_api\"h\n\x10InferenceRequest\x12\x11\n\tinput_ids\x18\x01 \x03(\x05\x12\x15\n\rinput_lengths\x18\x02 \x03(\x05\x12\x16\n\x0eoutput_lengths\x18\x03 \x03(\x05\x12\x12\n\nubatch_ids\x18\x04 \x03(\x05\"m\n\x16InferenceRequestUbatch\x12\x11\n\tubatch_id\x18\x01 \x01(\x05\x12\x11\n\tinput_ids\x18\x02 \x03(\x05\x12\x15\n\rinput_lengths\x18\x03 \x03(\x05\x12\x16\n\x0eoutput_lengths\x18\x04 \x03(\x05\"\x13\n\x11InferenceResponse\"\x17\n\x15InferenceResponseOpen2\xa9\x01\n\tApiServer\x12\x46\n\x05Serve\x12\x1c.dejavu_api.InferenceRequest\x1a\x1d.dejavu_api.InferenceResponse\"\x00\x12T\n\tServeOpen\x12\".dejavu_api.InferenceRequestUbatch\x1a!.dejavu_api.InferenceResponseOpen\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'protos.api_server_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_INFERENCEREQUEST']._serialized_start=39
  _globals['_INFERENCEREQUEST']._serialized_end=143
  _globals['_INFERENCEREQUESTUBATCH']._serialized_start=145
  _globals['_INFERENCEREQUESTUBATCH']._serialized_end=254
  _globals['_INFERENCERESPONSE']._serialized_start=256
  _globals['_INFERENCERESPONSE']._serialized_end=275
  _globals['_INFERENCERESPONSEOPEN']._serialized_start=277
  _globals['_INFERENCERESPONSEOPEN']._serialized_end=300
  _globals['_APISERVER']._serialized_start=303
  _globals['_APISERVER']._serialized_end=472
# @@protoc_insertion_point(module_scope)
