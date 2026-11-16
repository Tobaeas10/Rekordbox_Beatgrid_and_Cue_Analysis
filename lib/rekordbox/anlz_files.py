"""Read and rebuild the PQTZ beat grid of a Rekordbox ANLZ .DAT file (raw bytes, big endian).
Layout: file header "PMAI" (len_header @4, len_file @8), then tags [type 4][len_header 4][len_tag 4].
PQTZ: len_header 24, bytes 12..19 kept as found, entry_count u32 @20, entries from 24, 8 bytes each (beat u16, tempo u16 = bpm*100, time u32 ms).
Pure functions, never touch files. The extended PQT2 tag of .EXT files is not handled here."""
import struct


def iter_tags(data):
    offset = struct.unpack_from(">I", data, 4)[0]
    while offset + 12 <= len(data):
        tag_type = bytes(data[offset:offset + 4]).decode("ascii", "replace")
        header_length, tag_length = struct.unpack_from(">II", data, offset + 4)
        if tag_length < 12:
            break
        yield tag_type, offset, header_length, tag_length
        offset += tag_length


def read_pqtz(data):
    for tag_type, offset, header_length, tag_length in iter_tags(data):
        if tag_type == "PQTZ":
            entry_count = struct.unpack_from(">I", data, offset + 20)[0]
            return [struct.unpack_from(">HHI", data, offset + 24 + 8 * entry_index) for entry_index in range(entry_count)]
    return []


def check_structure(data):
    """True when the header file length and the tag chain agree with the byte length."""
    if bytes(data[:4]) != b"PMAI" or struct.unpack_from(">I", data, 8)[0] != len(data):
        return False
    end = 0
    for tag_type, offset, header_length, tag_length in iter_tags(data):
        end = offset + tag_length
    return end == len(data)


def set_pqtz(data, beats):
    """Return new bytes with the PQTZ entries replaced by beats = [(beat_in_bar, tempo_x100, ms), ...].
    All other tags stay byte-identical; PQTZ tag length and the file length in the header are updated."""
    for tag_type, offset, header_length, tag_length in iter_tags(data):
        if tag_type != "PQTZ":
            continue
        head = bytearray(data[offset:offset + 24])
        struct.pack_into(">I", head, 8, 24 + 8 * len(beats))
        struct.pack_into(">I", head, 20, len(beats))
        body = b"".join(struct.pack(">HHI", beat_in_bar, tempo_times_100, milliseconds) for beat_in_bar, tempo_times_100, milliseconds in beats)
        out = bytearray(data[:offset]) + head + body + bytearray(data[offset + tag_length:])
        struct.pack_into(">I", out, 8, len(out))
        return bytes(out)
    raise ValueError("no PQTZ tag")


def roundtrip_ok(data, beats):
    """Dry run of the write: rebuild, parse back, compare, check structure and that other tags are untouched."""
    new = set_pqtz(data, beats)
    same_other = [(tag_type, bytes(data[tag_offset:tag_offset + tag_length])) for tag_type, tag_offset, header_length, tag_length in iter_tags(data) if tag_type != "PQTZ"] == \
                 [(tag_type, bytes(new[tag_offset:tag_offset + tag_length])) for tag_type, tag_offset, header_length, tag_length in iter_tags(new) if tag_type != "PQTZ"]
    return check_structure(new) and [tuple(beat) for beat in read_pqtz(new)] == [tuple(beat) for beat in beats] and same_other


def cue_entries(data, tag_type, offset, header_length):
    """Yield (time_offset, loop_offset) absolute offsets of cue entries in a PCOB/PCO2 tag."""
    if tag_type == "PCOB":
        entry_count, time_field_offset, loop_field_offset = struct.unpack_from(">H", data, offset + 18)[0], 32, 36
    else:
        entry_count, time_field_offset, loop_field_offset = struct.unpack_from(">H", data, offset + 16)[0], 20, 24
    entry_position = offset + header_length
    for _ in range(entry_count):
        yield entry_position + time_field_offset, entry_position + loop_field_offset
        entry_position += struct.unpack_from(">I", data, entry_position + 8)[0]
