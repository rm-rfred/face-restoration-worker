# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: face_restoration.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16\x66\x61\x63\x65_restoration.proto\x12\x10\x66\x61\x63\x65_restoration\"\x88\x01\n\x16\x46\x61\x63\x65RestorationRequest\x12\r\n\x05image\x18\x01 \x01(\x0c\x12\x1a\n\x12\x62\x61\x63kground_enhance\x18\x02 \x01(\x08\x12\x15\n\rface_upsample\x18\x03 \x01(\x08\x12\x0f\n\x07upscale\x18\x04 \x01(\x05\x12\x1b\n\x13\x63odeformer_fidelity\x18\x05 \x01(\x02\".\n\x14\x46\x61\x63\x65RestorationReply\x12\x16\n\x0erestored_image\x18\x01 \x01(\x0c\x32\x84\x01\n\x16\x46\x61\x63\x65RestorationService\x12j\n\x14\x41pplyFaceRestoration\x12(.face_restoration.FaceRestorationRequest\x1a&.face_restoration.FaceRestorationReply\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'face_restoration_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_FACERESTORATIONREQUEST']._serialized_start=45
  _globals['_FACERESTORATIONREQUEST']._serialized_end=181
  _globals['_FACERESTORATIONREPLY']._serialized_start=183
  _globals['_FACERESTORATIONREPLY']._serialized_end=229
  _globals['_FACERESTORATIONSERVICE']._serialized_start=232
  _globals['_FACERESTORATIONSERVICE']._serialized_end=364
# @@protoc_insertion_point(module_scope)