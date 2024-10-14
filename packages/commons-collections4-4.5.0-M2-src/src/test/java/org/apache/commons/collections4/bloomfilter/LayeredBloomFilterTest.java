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
package org.apache.commons.collections4.bloomfilter;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.function.Supplier;

import org.apache.commons.collections4.bloomfilter.LayerManager.Cleanup;
import org.apache.commons.collections4.bloomfilter.LayerManager.ExtendCheck;
import org.junit.jupiter.api.Test;

public class LayeredBloomFilterTest extends AbstractBloomFilterTest<LayeredBloomFilter<?>> {

    /**
     * A Predicate that advances after a quantum of time.
     */
    static class AdvanceOnTimeQuanta implements Predicate<LayerManager<TimestampedBloomFilter>> {
        long quanta;

        AdvanceOnTimeQuanta(final long quanta, final TimeUnit unit) {
            this.quanta = unit.toMillis(quanta);
        }

        @Override
        public boolean test(final LayerManager<TimestampedBloomFilter> lm) {
            // can not use getTarget() as it causes recursion.
            return lm.last().timestamp + quanta < System.currentTimeMillis();
        }
    }

    /**
     * A Consumer that cleans the list based on how long each filters has been in
     * the list.
     */
    static class CleanByTime<T extends TimestampedBloomFilter> implements Consumer<List<T>> {
        long elapsedTime;

        CleanByTime(final long duration, final TimeUnit unit) {
            elapsedTime = unit.toMillis(duration);
        }

        @Override
        public void accept(final List<T> t) {
            final long min = System.currentTimeMillis() - elapsedTime;
            final Iterator<T> iter = t.iterator();
            while (iter.hasNext()) {
                final TimestampedBloomFilter bf = iter.next();
                if (bf.getTimestamp() >= min) {
                    return;
                }
                dbgInstrument.add(String.format("Removing old entry: T:%s (Aged: %s) \n", bf.getTimestamp(),
                        min - bf.getTimestamp()));
                iter.remove();
            }
        }
    }

    static class NumberedBloomFilter extends WrappedBloomFilter {
        int value;
        int sequence;
        NumberedBloomFilter(final Shape shape, final int value, final int sequence) {
            super(new SimpleBloomFilter(shape));
            this.value = value;
            this.sequence = sequence;
        }

        @Override
        public BloomFilter copy() {
            return new NumberedBloomFilter(getShape(), value, sequence);
        }
    }

    /**
     * A Bloomfilter implementation that tracks the creation time.
     */
    public static class TimestampedBloomFilter extends WrappedBloomFilter {
        final long timestamp;

        TimestampedBloomFilter(final BloomFilter bf) {
            super(bf);
            this.timestamp = System.currentTimeMillis();
        }

        TimestampedBloomFilter(final BloomFilter bf, final long timestamp) {
            super(bf);
            this.timestamp = timestamp;
        }

        @Override
        public TimestampedBloomFilter copy() {
            return new TimestampedBloomFilter(this.getWrapped().copy(), timestamp);
        }

        public long getTimestamp() {
            return timestamp;
        }
    }

    // ***example of instrumentation ***
    private static final List<String> dbgInstrument = new ArrayList<>();

    /**
     * Creates a LayeredBloomFilter that retains enclosed filters for
     * {@code duration} and limits the contents of each enclosed filter to a time
     * {@code quanta}. This filter uses the timestamped Bloom filter internally.
     *
     * @param shape    The shape of the Bloom filters.
     * @param duration The length of time to keep filters in the list.
     * @param dUnit    The unit of time to apply to duration.
     * @param quanta   The quantization factor for each filter. Individual filters
     *                 will span at most this much time.
     * @param qUnit    the unit of time to apply to quanta.
     * @return LayeredBloomFilter with the above properties.
     */
    static LayeredBloomFilter<TimestampedBloomFilter> createTimedLayeredFilter(final Shape shape, final long duration, final TimeUnit dUnit, final long quanta,
            final TimeUnit qUnit) {
        final LayerManager.Builder<TimestampedBloomFilter> builder = LayerManager.builder();
        final Consumer<Deque<TimestampedBloomFilter>> cleanup = Cleanup.removeEmptyTarget().andThen(new CleanByTime(duration, dUnit));
        final LayerManager<TimestampedBloomFilter> layerManager = builder
                .setSupplier(() -> new TimestampedBloomFilter(new SimpleBloomFilter(shape)))
                .setCleanup(cleanup)
                .setExtendCheck(new AdvanceOnTimeQuanta(quanta, qUnit)
                        .or(LayerManager.ExtendCheck.advanceOnSaturation(shape.estimateMaxN())))
                .build();
        return new LayeredBloomFilter<>(shape, layerManager);
    }

    /**
     * Creates a fixed size layered bloom filter that adds new filters to the list,
     * but never merges them. List will never exceed maxDepth. As additional filters
     * are added earlier filters are removed.  Uses SimpleBloomFilters.
     *
     * @param shape    The shape for the enclosed Bloom filters.
     * @param maxDepth The maximum depth of layers.
     * @return An empty layered Bloom filter of the specified shape and depth.
     */
    public static  LayeredBloomFilter<BloomFilter> fixed(final Shape shape, final int maxDepth) {
        return fixed(shape, maxDepth, () -> new SimpleBloomFilter(shape));
    }

    /**
     * Creates a fixed size layered bloom filter that adds new filters to the list,
     * but never merges them. List will never exceed maxDepth. As additional filters
     * are added earlier filters are removed.
     *
     * @param shape    The shape for the enclosed Bloom filters.
     * @param maxDepth The maximum depth of layers.
     * @param supplier A supplier of the Bloom filters to create layers with.
     * @return An empty layered Bloom filter of the specified shape and depth.
     */
    public static <T extends BloomFilter> LayeredBloomFilter<T> fixed(final Shape shape, final int maxDepth, final Supplier<T> supplier) {
        final LayerManager.Builder<T> builder = LayerManager.builder();
        builder.setExtendCheck(LayerManager.ExtendCheck.advanceOnPopulated())
                .setCleanup(LayerManager.Cleanup.onMaxSize(maxDepth)).setSupplier(supplier);
        return new LayeredBloomFilter<>(shape, builder.build());
    }

    // instrumentation to record timestamps in dbgInstrument list
    private final Predicate<BloomFilter> dbg = bf -> {
        final TimestampedBloomFilter tbf = (TimestampedBloomFilter) bf;
        final long ts = System.currentTimeMillis();
        dbgInstrument.add(String.format("T:%s (Elapsed:%s)- EstN:%s (Card:%s)\n", tbf.timestamp, ts - tbf.timestamp,
                tbf.estimateN(), tbf.cardinality()));
        return true;
    };
    // *** end of instrumentation ***

    @Override
    protected LayeredBloomFilter<BloomFilter> createEmptyFilter(final Shape shape) {
        return LayeredBloomFilterTest.fixed(shape, 10);
    }

    protected BloomFilter makeFilter(final Hasher h) {
        final BloomFilter bf = new SparseBloomFilter(getTestShape());
        bf.merge(h);
        return bf;
    }

    protected BloomFilter makeFilter(final IndexExtractor p) {
        final BloomFilter bf = new SparseBloomFilter(getTestShape());
        bf.merge(p);
        return bf;
    }

    protected BloomFilter makeFilter(final int... values) {
        return makeFilter(IndexExtractor.fromIndexArray(values));
    }

    private LayeredBloomFilter<BloomFilter> setupFindTest() {
        final LayeredBloomFilter<BloomFilter> filter = LayeredBloomFilterTest.fixed(getTestShape(), 10);
        filter.merge(TestingHashers.FROM1);
        filter.merge(TestingHashers.FROM11);
        filter.merge(new IncrementingHasher(11, 2));
        filter.merge(TestingHashers.populateFromHashersFrom1AndFrom11(new SimpleBloomFilter(getTestShape())));
        return filter;
    }

    @Override
    @Test
    public void testCardinalityAndIsEmpty() {
        final LayerManager<BloomFilter> layerManager = LayerManager.builder().setExtendCheck(ExtendCheck.neverAdvance())
                .setSupplier(() -> new SimpleBloomFilter(getTestShape())).build();
        testCardinalityAndIsEmpty(new LayeredBloomFilter<>(getTestShape(), layerManager));
    }

    // ***** TESTS THAT CHECK LAYERED PROCESSING ******

    @Test
    public void testCleanup() {
        final int[] sequence = {1};
        final LayerManager layerManager = LayerManager.builder()
                .setSupplier(() -> new NumberedBloomFilter(getTestShape(), 3, sequence[0]++))
                .setExtendCheck(ExtendCheck.neverAdvance())
                .setCleanup(ll -> ll.removeIf( f -> (((NumberedBloomFilter) f).value-- == 0))).build();
        final LayeredBloomFilter underTest = new LayeredBloomFilter(getTestShape(), layerManager );
        assertEquals(1, underTest.getDepth());
        underTest.merge(TestingHashers.randomHasher());
        underTest.cleanup(); // first count == 2
        assertEquals(1, underTest.getDepth());
        underTest.next(); // first count == 1
        assertEquals(2, underTest.getDepth());
        underTest.merge(TestingHashers.randomHasher());
        underTest.cleanup(); // first count == 0
        NumberedBloomFilter f = (NumberedBloomFilter) underTest.get(0);
        assertEquals(1, f.sequence);

        assertEquals(2, underTest.getDepth());
        underTest.cleanup(); // should be removed ; second is now 1st with value 1
        assertEquals(1, underTest.getDepth());
        f = (NumberedBloomFilter) underTest.get(0);
        assertEquals(2, f.sequence);

        underTest.cleanup(); // first count == 0
        underTest.cleanup(); // should be removed.  But there is always at least one
        assertEquals(1, underTest.getDepth());
        f = (NumberedBloomFilter) underTest.get(0);
        assertEquals(3, f.sequence);  // it is a new one.
    }
    /**
     * Tests that the estimated union calculations are correct.
     */
    @Test
    public final void testEstimateUnionCrossTypes() {
        final BloomFilter bf = createFilter(getTestShape(), TestingHashers.FROM1);
        final BloomFilter bf2 = new DefaultBloomFilterTest.SparseDefaultBloomFilter(getTestShape());
        bf2.merge(TestingHashers.FROM11);

        assertEquals(2, bf.estimateUnion(bf2));
        assertEquals(2, bf2.estimateUnion(bf));
    }

    @Test
    public void testExpiration() throws InterruptedException {
        // this test uses the instrumentation noted above to track changes for debugging
        // purposes.

        // list of timestamps that are expected to be expired.
        final List<Long> lst = new ArrayList<>();
        final Shape shape = Shape.fromNM(4, 64);

        // create a filter that removes filters that are 4 seconds old
        // and quantises time to 1 second intervals.
        final LayeredBloomFilter<TimestampedBloomFilter> underTest = createTimedLayeredFilter(shape, 600, TimeUnit.MILLISECONDS, 150,
                TimeUnit.MILLISECONDS);

        for (int i = 0; i < 10; i++) {
            underTest.merge(TestingHashers.randomHasher());
        }
        underTest.processBloomFilters(dbg.and(x -> lst.add(((TimestampedBloomFilter) x).timestamp)));
        assertTrue(underTest.getDepth() > 1);

        Thread.sleep(300);
        for (int i = 0; i < 10; i++) {
            underTest.merge(TestingHashers.randomHasher());
        }
        dbgInstrument.add("=== AFTER 300 milliseconds ====\n");
        underTest.processBloomFilters(dbg);

        Thread.sleep(150);
        for (int i = 0; i < 10; i++) {
            underTest.merge(TestingHashers.randomHasher());
        }
        dbgInstrument.add("=== AFTER 450 milliseconds ====\n");
        underTest.processBloomFilters(dbg);

        // sleep 200 milliseconds to ensure we cross the 600 millisecond boundary
        Thread.sleep(200);
        underTest.merge(TestingHashers.randomHasher());
        dbgInstrument.add("=== AFTER 600 milliseconds ====\n");
        assertTrue(underTest.processBloomFilters(dbg.and(x -> !lst.contains(((TimestampedBloomFilter) x).timestamp))),
                "Found filter that should have been deleted: " + dbgInstrument.get(dbgInstrument.size() - 1));
    }

    @Test
    public void testFindBitMapExtractor() {
        final LayeredBloomFilter<BloomFilter> filter = setupFindTest();

        IndexExtractor indexExtractor = TestingHashers.FROM1.indices(getTestShape());
        BitMapExtractor bitMapExtractor = BitMapExtractor.fromIndexExtractor(indexExtractor, getTestShape().getNumberOfBits());

        int[] expected = {0, 3};
        int[] result = filter.find(bitMapExtractor);
        assertArrayEquals(expected, result);

        expected = new int[]{1, 3};
        indexExtractor = TestingHashers.FROM11.indices(getTestShape());
        bitMapExtractor = BitMapExtractor.fromIndexExtractor(indexExtractor, getTestShape().getNumberOfBits());
        result = filter.find(bitMapExtractor);
        assertArrayEquals(expected, result);
    }

    @Test
    public void testFindBloomFilter() {
        final LayeredBloomFilter<BloomFilter> filter = setupFindTest();
        int[] expected = {0, 3};
        int[] result = filter.find(TestingHashers.FROM1);
        assertArrayEquals(expected, result);
        expected = new int[] {1, 3};
        result = filter.find(TestingHashers.FROM11);
        assertArrayEquals(expected, result);
    }

    @Test
    public void testFindIndexExtractor() {
        IndexExtractor indexExtractor = TestingHashers.FROM1.indices(getTestShape());
        final LayeredBloomFilter<BloomFilter> filter = setupFindTest();

        int[] expected = {0, 3};
        int[] result = filter.find(indexExtractor);
        assertArrayEquals(expected, result);

        expected = new int[] {1, 3};
        indexExtractor = TestingHashers.FROM11.indices(getTestShape());
        result = filter.find(indexExtractor);
        assertArrayEquals(expected, result);
    }

    @Test
    public final void testGetLayer() {
        final BloomFilter bf = new SimpleBloomFilter(getTestShape());
        bf.merge(TestingHashers.FROM11);
        final LayeredBloomFilter<BloomFilter> filter = LayeredBloomFilterTest.fixed(getTestShape(), 10);
        filter.merge(TestingHashers.FROM1);
        filter.merge(TestingHashers.FROM11);
        filter.merge(new IncrementingHasher(11, 2));
        filter.merge(TestingHashers.populateFromHashersFrom1AndFrom11(new SimpleBloomFilter(getTestShape())));
        assertArrayEquals(bf.asBitMapArray(), filter.get(1).asBitMapArray());
    }

    @Test
    public void testMultipleFilters() {
        final LayeredBloomFilter<BloomFilter> filter = LayeredBloomFilterTest.fixed(getTestShape(), 10);
        filter.merge(TestingHashers.FROM1);
        filter.merge(TestingHashers.FROM11);
        assertEquals(2, filter.getDepth());
        assertTrue(filter.contains(makeFilter(TestingHashers.FROM1)));
        assertTrue(filter.contains(makeFilter(TestingHashers.FROM11)));
        final BloomFilter t1 = makeFilter(6, 7, 17, 18, 19);
        assertFalse(filter.contains(t1));
        assertFalse(filter.copy().contains(t1));
        assertTrue(filter.flatten().contains(t1));
    }

    @Test
    public final void testNext() {
        final LayerManager<BloomFilter> layerManager = LayerManager.builder().setSupplier(() -> new SimpleBloomFilter(getTestShape()))
                .build();

        final LayeredBloomFilter<BloomFilter> filter = new LayeredBloomFilter<>(getTestShape(), layerManager);
        filter.merge(TestingHashers.FROM1);
        filter.merge(TestingHashers.FROM11);
        assertEquals(1, filter.getDepth());
        filter.next();
        filter.merge(new IncrementingHasher(11, 2));
        assertEquals(2, filter.getDepth());
        assertTrue(filter.get(0).contains(TestingHashers.FROM1));
        assertTrue(filter.get(0).contains(TestingHashers.FROM11));
        assertFalse(filter.get(0).contains(new IncrementingHasher(11, 2)));
        assertFalse(filter.get(1).contains(TestingHashers.FROM1));
        assertFalse(filter.get(1).contains(TestingHashers.FROM11));
        assertTrue(filter.get(1).contains(new IncrementingHasher(11, 2)));
    }
}
