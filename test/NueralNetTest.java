import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class NueralNetTest {

  @BeforeEach
  void setUp() throws Exception {
    int[] layer_sizes = { 784, 16, 16, 10 };
    NueralNet nn = new NueralNet(layer_sizes);
  }

  @AfterEach
  void tearDown() throws Exception {}

  @Test
  void testNueralNet() {
    fail("Not yet implemented");
  }

  @Test
  void testPredict() {
    fail("Not yet implemented");
  }

  @Test
  void testBackprop() {
    fail("Not yet implemented");
  }

  @Test
  void testApplyBackprop() {
    fail("Not yet implemented");
  }

}
