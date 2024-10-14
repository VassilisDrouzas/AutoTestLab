/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.commons.codec.binary;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import org.apache.commons.codec.CodecPolicy;
import org.apache.commons.codec.DecoderException;
import org.apache.commons.lang3.ArrayUtils;
import org.junit.jupiter.api.Test;

public class Base32Test {

    private static final Charset CHARSET_UTF8 = StandardCharsets.UTF_8;

    /** RFC 4648. */
    // @formatter:off
    private static final String [][] BASE32_TEST_CASES = {
        { ""       , "" },
        { "f"      , "MY======" },
        { "fo"     , "MZXQ====" },
        { "foo"    , "MZXW6===" },
        { "foob"   , "MZXW6YQ=" },
        { "fooba"  , "MZXW6YTB" },
        { "foobar" , "MZXW6YTBOI======" }
    };
    // @formatter:on

    /**
     * Example test cases with valid characters but impossible combinations of
     * trailing characters (i.e. cannot be created during encoding).
     */
    // @formatter:off
    static final String[] BASE32_IMPOSSIBLE_CASES = {
        "MC======",
        "MZXE====",
        "MZXWB===",
        "MZXW6YB=",
        "MZXW6YTBOC======",
        "AB======"
        };
    // @formatter:on

    // @formatter:off
    private static final String[] BASE32_IMPOSSIBLE_CASES_CHUNKED = {
        "M2======\r\n",
        "MZX0====\r\n",
        "MZXW0===\r\n",
        "MZXW6Y2=\r\n",
        "MZXW6YTBO2======\r\n"
    };
    // @formatter:on

    // @formatter:off
    private static final String[] BASE32HEX_IMPOSSIBLE_CASES = {
        "C2======",
        "CPN4====",
        "CPNM1===",
        "CPNMUO1=",
        "CPNMUOJ1E2======"
    };
    // @formatter:on

    /**
     * Copy of the standard base-32 encoding table. Used to test decoding the final
     * character of encoded bytes.
     */
    // @formatter:off
    private static final byte[] ENCODE_TABLE = {
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '2', '3', '4', '5', '6', '7'
    };
    // @formatter:on

    private static final Object[][] BASE32_BINARY_TEST_CASES;

//            { null, "O0o0O0o0" }
//            BASE32_BINARY_TEST_CASES[2][0] = new Hex().decode("739ce739ce");

    static {
        final Hex hex = new Hex();
        try {
            BASE32_BINARY_TEST_CASES = new Object[][] {
                    new Object[] { hex.decode("623a01735836e9a126e12fbf95e013ee6892997c"),
                                   "MI5AC42YG3U2CJXBF67ZLYAT5ZUJFGL4" },
                    new Object[] { hex.decode("623a01735836e9a126e12fbf95e013ee6892997c"),
                                   "mi5ac42yg3u2cjxbf67zlyat5zujfgl4" },
                    new Object[] { hex.decode("739ce42108"),
                                   "OOOOIIII" }
            };
        } catch (final DecoderException de) {
            throw new AssertionError(":(", de);
        }
    }

    // @formatter:off
    private static final String [][] BASE32HEX_TEST_CASES = { // RFC 4648
        { ""       , "" },
        { "f"      , "CO======" },
        { "fo"     , "CPNG====" },
        { "foo"    , "CPNMU===" },
        { "foob"   , "CPNMUOG=" },
        { "fooba"  , "CPNMUOJ1" },
        { "foobar" , "CPNMUOJ1E8======" }
    };
    // @formatter:on

    // @formatter:off
    private static final String [][] BASE32_TEST_CASES_CHUNKED = { //Chunked
        { ""       , "" },
        { "f"      , "MY======\r\n" },
        { "fo"     , "MZXQ====\r\n" },
        { "foo"    , "MZXW6===\r\n" },
        { "foob"   , "MZXW6YQ=\r\n" },
        { "fooba"  , "MZXW6YTB\r\n" },
        { "foobar" , "MZXW6YTBOI======\r\n" }
    };
    // @formatter:on

    // @formatter:off
    private static final String [][] BASE32_PAD_TEST_CASES = { // RFC 4648
        { ""       , "" },
        { "f"      , "MY%%%%%%" },
        { "fo"     , "MZXQ%%%%" },
        { "foo"    , "MZXW6%%%" },
        { "foob"   , "MZXW6YQ%" },
        { "fooba"  , "MZXW6YTB" },
        { "foobar" , "MZXW6YTBOI%%%%%%" }
    };
    // @formatter:on

    /**
     * Test base 32 decoding of the final trailing bits. Trailing encoded bytes
     * cannot fit exactly into 5-bit characters so the last character has a limited
     * alphabet where the final bits are zero. This asserts that illegal final
     * characters throw an exception when decoding.
     *
     * @param nbits the number of trailing bits (must be a factor of 5 and {@code <40})
     */
    private static void assertBase32DecodingOfTrailingBits(final int nbits) {
        // Requires strict decoding
        final Base32 codec = new Base32(0, null, false, BaseNCodec.PAD_DEFAULT, CodecPolicy.STRICT);
        assertTrue(codec.isStrictDecoding());
        assertEquals(CodecPolicy.STRICT, codec.getCodecPolicy());
        // A lenient decoder should not re-encode to the same bytes
        final Base32 defaultCodec = new Base32();
        assertFalse(defaultCodec.isStrictDecoding());
        assertEquals(CodecPolicy.LENIENT, defaultCodec.getCodecPolicy());

        // Create the encoded bytes. The first characters must be valid so fill with 'zero'
        // then pad to the block size.
        final int length = nbits / 5;
        final byte[] encoded = new byte[8];
        Arrays.fill(encoded, 0, length, ENCODE_TABLE[0]);
        Arrays.fill(encoded, length, encoded.length, (byte) '=');
        // Compute how many bits would be discarded from 8-bit bytes
        final int discard = nbits % 8;
        final int emptyBitsMask = (1 << discard) - 1;
        // Special case when an impossible number of trailing characters
        final boolean invalid = length == 1 || length == 3 || length == 6;
        // Enumerate all 32 possible final characters in the last position
        final int last = length - 1;
        for (int i = 0; i < 32; i++) {
            encoded[last] = ENCODE_TABLE[i];
            // If the lower bits are set we expect an exception. This is not a valid
            // final character.
            if (invalid || (i & emptyBitsMask) != 0) {
                assertThrows(IllegalArgumentException.class, () -> codec.decode(encoded), "Final base-32 digit should not be allowed");
                // The default lenient mode should decode this
                final byte[] decoded = defaultCodec.decode(encoded);
                // Re-encoding should not match the original array as it was invalid
                assertFalse(Arrays.equals(encoded, defaultCodec.encode(decoded)));
            } else {
                // Otherwise this should decode
                final byte[] decoded = codec.decode(encoded);
                // Compute the bits that were encoded. This should match the final decoded byte.
                final int bitsEncoded = i >> discard;
                assertEquals(bitsEncoded, decoded[decoded.length - 1], "Invalid decoding of last character");
                // Re-encoding should match the original array (requires the same padding character)
                assertArrayEquals(encoded, codec.encode(decoded));
            }
        }
    }

    @Test
    public void testBase32AtBufferEnd() {
        testBase32InBuffer(100, 0);
    }

    @Test
    public void testBase32AtBufferMiddle() {
        testBase32InBuffer(100, 100);
    }

    @Test
    public void testBase32AtBufferStart() {
        testBase32InBuffer(0, 100);
    }

    @Test
    public void testBase32BinarySamples() throws Exception {
        final Base32 codec = new Base32();
        for (final Object[] element : BASE32_BINARY_TEST_CASES) {
            final String expected;
            if (element.length > 2) {
                expected = (String) element[2];
            } else {
                expected = (String) element[1];
            }
            assertEquals(expected.toUpperCase(), codec.encodeAsString((byte[]) element[0]));
        }
    }

    @Test
    public void testBase32BinarySamplesReverse() throws Exception {
        final Base32 codec = new Base32();
        for (final Object[] element : BASE32_BINARY_TEST_CASES) {
            assertArrayEquals((byte[]) element[0], codec.decode((String) element[1]));
        }
    }

    @Test
    public void testBase32Chunked() throws Exception {
        final Base32 codec = new Base32(20);
        for (final String[] element : BASE32_TEST_CASES_CHUNKED) {
            assertEquals(element[1], codec.encodeAsString(element[0].getBytes(CHARSET_UTF8)));
        }
    }

    @Test
    public void testBase32DecodingOfTrailing10Bits() {
        assertBase32DecodingOfTrailingBits(10);
    }

    @Test
    public void testBase32DecodingOfTrailing15Bits() {
        assertBase32DecodingOfTrailingBits(15);
    }

    @Test
    public void testBase32DecodingOfTrailing20Bits() {
        assertBase32DecodingOfTrailingBits(20);
    }

    @Test
    public void testBase32DecodingOfTrailing25Bits() {
        assertBase32DecodingOfTrailingBits(25);
    }

    @Test
    public void testBase32DecodingOfTrailing30Bits() {
        assertBase32DecodingOfTrailingBits(30);
    }

    @Test
    public void testBase32DecodingOfTrailing35Bits() {
        assertBase32DecodingOfTrailingBits(35);
    }

    @Test
    public void testBase32DecodingOfTrailing5Bits() {
        assertBase32DecodingOfTrailingBits(5);
    }

    @Test
    public void testBase32HexImpossibleSamples() {
        testImpossibleCases(new Base32(0, null, true, BaseNCodec.PAD_DEFAULT, CodecPolicy.STRICT), BASE32HEX_IMPOSSIBLE_CASES);
    }

    @Test
    public void testBase32HexSamples() throws Exception {
        final Base32 codec = new Base32(true);
        for (final String[] element : BASE32HEX_TEST_CASES) {
            assertEquals(element[1], codec.encodeAsString(element[0].getBytes(CHARSET_UTF8)));
        }
    }

    @Test
    public void testBase32HexSamplesReverse() throws Exception {
        final Base32 codec = new Base32(true);
        for (final String[] element : BASE32HEX_TEST_CASES) {
            assertEquals(element[0], new String(codec.decode(element[1]), CHARSET_UTF8));
        }
    }

    @Test
    public void testBase32HexSamplesReverseLowercase() throws Exception {
        final Base32 codec = new Base32(true);
        for (final String[] element : BASE32HEX_TEST_CASES) {
            assertEquals(element[0], new String(codec.decode(element[1].toLowerCase()), CHARSET_UTF8));
        }
    }

    @Test
    public void testBase32ImpossibleChunked() {
        testImpossibleCases(new Base32(20, BaseNCodec.CHUNK_SEPARATOR, false, BaseNCodec.PAD_DEFAULT, CodecPolicy.STRICT), BASE32_IMPOSSIBLE_CASES_CHUNKED);
    }

    @Test
    public void testBase32ImpossibleSamples() {
        testImpossibleCases(new Base32(0, null, false, BaseNCodec.PAD_DEFAULT, CodecPolicy.STRICT), BASE32_IMPOSSIBLE_CASES);
    }

    private void testBase32InBuffer(final int startPasSize, final int endPadSize) {
        final Base32 codec = new Base32();
        for (final String[] element : BASE32_TEST_CASES) {
            final byte[] bytes = element[0].getBytes(CHARSET_UTF8);
            byte[] buffer = ArrayUtils.addAll(bytes, new byte[endPadSize]);
            buffer = ArrayUtils.addAll(new byte[startPasSize], buffer);
            assertEquals(element[1], StringUtils.newStringUtf8(codec.encode(buffer, startPasSize, bytes.length)));
        }
    }

    @Test
    public void testBase32Samples() throws Exception {
        final Base32 codec = new Base32();
        for (final String[] element : BASE32_TEST_CASES) {
            assertEquals(element[1], codec.encodeAsString(element[0].getBytes(CHARSET_UTF8)));
        }
    }

    @Test
    public void testBase32SamplesNonDefaultPadding() throws Exception {
        final Base32 codec = new Base32((byte) 0x25); // '%' <=> 0x25

        for (final String[] element : BASE32_PAD_TEST_CASES) {
            assertEquals(element[1], codec.encodeAsString(element[0].getBytes(CHARSET_UTF8)));
        }
    }

    @Test
    public void testBuilderCodecPolicy() {
        assertEquals(CodecPolicy.LENIENT, Base32.builder().get().getCodecPolicy());
        assertEquals(CodecPolicy.LENIENT, Base32.builder().setDecodingPolicy(CodecPolicy.LENIENT).get().getCodecPolicy());
        assertEquals(CodecPolicy.STRICT, Base32.builder().setDecodingPolicy(CodecPolicy.STRICT).get().getCodecPolicy());
        assertEquals(CodecPolicy.LENIENT, Base32.builder().setDecodingPolicy(CodecPolicy.STRICT).setDecodingPolicy(null).get().getCodecPolicy());
        assertEquals(CodecPolicy.LENIENT, Base32.builder().setDecodingPolicy(null).get().getCodecPolicy());
    }

    @Test
    public void testBuilderLineAttributes() {
        assertNull(Base32.builder().get().getLineSeparator());
        assertNull(Base32.builder().setLineSeparator(BaseNCodec.CHUNK_SEPARATOR).get().getLineSeparator());
        assertArrayEquals(BaseNCodec.CHUNK_SEPARATOR, Base32.builder().setLineLength(4).setLineSeparator(BaseNCodec.CHUNK_SEPARATOR).get().getLineSeparator());
        assertArrayEquals(BaseNCodec.CHUNK_SEPARATOR, Base32.builder().setLineLength(4).setLineSeparator(null).get().getLineSeparator());
        assertArrayEquals(BaseNCodec.CHUNK_SEPARATOR, Base32.builder().setLineLength(10).setLineSeparator(null).get().getLineSeparator());
        assertNull(Base32.builder().setLineLength(-1).setLineSeparator(null).get().getLineSeparator());
        assertNull(Base32.builder().setLineLength(0).setLineSeparator(null).get().getLineSeparator());
        assertArrayEquals(new byte[] { 1 }, Base32.builder().setLineLength(4).setLineSeparator((byte) 1).get().getLineSeparator());
        assertEquals("MZXXQ===", Base32.builder().setLineLength(4).get().encodeToString("fox".getBytes(CHARSET_UTF8)));
    }

    @Test
    public void testBuilderPadingByte() {
        assertNull(Base32.builder().get().getLineSeparator());
        assertNull(Base32.builder().setLineSeparator(BaseNCodec.CHUNK_SEPARATOR).get().getLineSeparator());
        assertArrayEquals(BaseNCodec.CHUNK_SEPARATOR, Base32.builder().setLineLength(4).setLineSeparator(BaseNCodec.CHUNK_SEPARATOR).get().getLineSeparator());
        assertArrayEquals(BaseNCodec.CHUNK_SEPARATOR, Base32.builder().setLineLength(4).setLineSeparator(null).get().getLineSeparator());
        assertArrayEquals(BaseNCodec.CHUNK_SEPARATOR, Base32.builder().setLineLength(10).setLineSeparator(null).get().getLineSeparator());
        assertNull(Base32.builder().setLineLength(-1).setLineSeparator(null).get().getLineSeparator());
        assertNull(Base32.builder().setLineLength(0).setLineSeparator(null).get().getLineSeparator());
        assertArrayEquals(new byte[] { 1 }, Base32.builder().setLineLength(4).setLineSeparator((byte) 1).get().getLineSeparator());
        assertEquals("MZXXQ___", Base32.builder().setLineLength(4).setPadding((byte) '_').get().encodeToString("fox".getBytes(CHARSET_UTF8)));
    }

    @Test
    public void testCodec200() {
        final Base32 codec = new Base32(true, (byte) 'W'); // should be allowed
        assertNotNull(codec);
    }

    @Test
    public void testConstructors() {
        Base32 base32;
        base32 = new Base32();
        base32 = new Base32(-1);
        base32 = new Base32(-1, new byte[] {});
        base32 = new Base32(32, new byte[] {});
        base32 = new Base32(32, new byte[] {}, false);
        // This is different behavior than Base64 which validates the separator
        // even when line length is negative.
        base32 = new Base32(-1, new byte[] { 'A' });
        base32 = new Base32(32, new byte[] { '$' }); // OK
        assertThrows(IllegalArgumentException.class, () -> new Base32(32, null), "null line separator");
        assertThrows(IllegalArgumentException.class, () -> new Base32(32, new byte[] { 'A' }), "'A' as a line separator");
        assertThrows(IllegalArgumentException.class, () -> new Base32(32, new byte[] { '=' }), "'=' as a line separator");
        assertThrows(IllegalArgumentException.class, () -> new Base32(32, new byte[] { 'A', '$' }), "'A$' as a line separator");
        assertThrows(IllegalArgumentException.class, () -> new Base32(32, new byte[] { '\n' }, false, (byte) 'A'), "'A' as padding");
        assertThrows(IllegalArgumentException.class, () -> new Base32(32, new byte[] { '\n' }, false, (byte) ' '), "' ' as padding");

        base32 = new Base32(32, new byte[] { ' ', '$', '\n', '\r', '\t' }); // OK
        assertNotNull(base32);
    }

    /**
     * Test encode and decode of empty byte array.
     */
    @Test
    public void testEmptyBase32() {
        byte[] empty = {};
        byte[] result = new Base32().encode(empty);
        assertEquals(0, result.length, "empty Base32 encode");
        assertNull(new Base32().encode(null), "empty Base32 encode");
        result = new Base32().encode(empty, 0, 1);
        assertEquals(0, result.length, "empty Base32 encode with offset");
        assertNull(new Base32().encode(null), "empty Base32 encode with offset");

        empty = new byte[0];
        result = new Base32().decode(empty);
        assertEquals(0, result.length, "empty Base32 decode");
        assertNull(new Base32().decode((byte[]) null), "empty Base32 encode");
    }

    private void testImpossibleCases(final Base32 codec, final String[] impossible_cases) {
        for (final String impossible : impossible_cases) {
            assertThrows(IllegalArgumentException.class, () -> codec.decode(impossible));
        }
    }

    @Test
    public void testIsInAlphabet() {
        // invalid bounds
        Base32 b32 = new Base32(true);
        assertFalse(b32.isInAlphabet((byte) 0));
        assertFalse(b32.isInAlphabet((byte) 1));
        assertFalse(b32.isInAlphabet((byte) -1));
        assertFalse(b32.isInAlphabet((byte) -15));
        assertFalse(b32.isInAlphabet((byte) -32));
        assertFalse(b32.isInAlphabet((byte) 127));
        assertFalse(b32.isInAlphabet((byte) 128));
        assertFalse(b32.isInAlphabet((byte) 255));

        // default table
        b32 = new Base32(false);
        for (char c = '2'; c <= '7'; c++) {
            assertTrue(b32.isInAlphabet((byte) c));
        }
        for (char c = 'A'; c <= 'Z'; c++) {
            assertTrue(b32.isInAlphabet((byte) c));
        }
        for (char c = 'a'; c <= 'z'; c++) {
            assertTrue(b32.isInAlphabet((byte) c));
        }
        assertFalse(b32.isInAlphabet((byte) '1'));
        assertFalse(b32.isInAlphabet((byte) '8'));
        assertFalse(b32.isInAlphabet((byte) ('A' - 1)));
        assertFalse(b32.isInAlphabet((byte) ('Z' + 1)));

        // hex table
        b32 = new Base32(true);
        for (char c = '0'; c <= '9'; c++) {
            assertTrue(b32.isInAlphabet((byte) c));
        }
        for (char c = 'A'; c <= 'V'; c++) {
            assertTrue(b32.isInAlphabet((byte) c));
        }
        for (char c = 'a'; c <= 'v'; c++) {
            assertTrue(b32.isInAlphabet((byte) c));
        }
        assertFalse(b32.isInAlphabet((byte) ('0' - 1)));
        assertFalse(b32.isInAlphabet((byte) ('9' + 1)));
        assertFalse(b32.isInAlphabet((byte) ('A' - 1)));
        assertFalse(b32.isInAlphabet((byte) ('V' + 1)));
        assertFalse(b32.isInAlphabet((byte) ('a' - 1)));
        assertFalse(b32.isInAlphabet((byte) ('v' + 1)));
    }

    @Test
    public void testRandomBytes() {
        for (int i = 0; i < 20; i++) {
            final Base32 codec = new Base32();
            final byte[][] b = BaseNTestData.randomData(codec, i);
            assertEquals(b[1].length, codec.getEncodedLength(b[0]), i + " " + codec.lineLength);
            // assertEquals(b[0], codec.decode(b[1]));
        }
    }

    @Test
    public void testRandomBytesChunked() {
        for (int i = 0; i < 20; i++) {
            final Base32 codec = new Base32(10);
            final byte[][] b = BaseNTestData.randomData(codec, i);
            assertEquals(b[1].length, codec.getEncodedLength(b[0]), i + " " + codec.lineLength);
            // assertEquals(b[0], codec.decode(b[1]));
        }
    }

    @Test
    public void testRandomBytesHex() {
        for (int i = 0; i < 20; i++) {
            final Base32 codec = new Base32(true);
            final byte[][] b = BaseNTestData.randomData(codec, i);
            assertEquals(b[1].length, codec.getEncodedLength(b[0]), i + " " + codec.lineLength);
            // assertEquals(b[0], codec.decode(b[1]));
        }
    }

    @Test
    public void testSingleCharEncoding() {
        for (int i = 0; i < 20; i++) {
            Base32 codec = new Base32();
            final BaseNCodec.Context context = new BaseNCodec.Context();
            final byte[] unencoded = new byte[i];
            final byte[] allInOne = codec.encode(unencoded);
            codec = new Base32();
            for (int j = 0; j < unencoded.length; j++) {
                codec.encode(unencoded, j, 1, context);
            }
            codec.encode(unencoded, 0, -1, context);
            final byte[] singly = new byte[allInOne.length];
            codec.readResults(singly, 0, 100, context);
            if (!Arrays.equals(allInOne, singly)) {
                fail();
            }
        }
    }
}
