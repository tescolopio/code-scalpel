public class HelloWorld {
    public static void main(String[] args) {
        int x = 10;
        int y = 20;
        
        if (x > 5) {
            int result = x + y;
            System.out.println(result);
        } else {
            int result = x - y;
            System.out.println(result);
        }
    }
    
    public int add(int a, int b) {
        return a + b;
    }
    
    public boolean isPositive(int n) {
        if (n > 0) {
            return true;
        }
        return false;
    }
}
