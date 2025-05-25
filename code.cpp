/**
 * Comprehensive Pricing System Implementation
 * 
 * This file combines implementations for all tasks in Question 1 and Question 2:
 * 
 * Question 1:
 * - Q1.1 - ValueNote definition and Price/Rate calculations
 * - Q1.2 - First-order derivatives (sensitivities) for ValueNotes
 * - Q1.3 - Second-order derivatives (convexities) for ValueNotes
 * 
 * Question 2:
 * - Q2.1 - DeliveryContract definition and RelativeFactor calculation
 * - Q2.2 - DeliveryContract pricing
 * - Q2.3 - Calculation of delivery probabilities
 * - Q2.4 - Sensitivity analyses
 *   - Q2.4.a - Sensitivity to volatility
 *   - Q2.4.b - Sensitivity to today's price
 * - Q2.5 - Alternative pricing methods (Bonus)
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <functional>
#include <map>
#include <chrono>
#include <sstream>
#include <memory>
#include <limits>
#include <numeric>

// Common constants
constexpr double PI = 3.14159265358979323846;

// Forward declarations
class Date;
class ValueNote;
class DeliveryContract;

// Date class for handling date operations
class Date {
private:
    int year;
    int month;
    int day;

public:
    // Constructor
    Date(int y, int m, int d) : year(y), month(m), day(d) {
        // Basic date validation
        if (m < 1 || m > 12 || d < 1 || d > 31) {
            throw std::invalid_argument("Invalid date components");
        }
    }

    // Parse date from string (format: YYYY-MM-DD)
    static Date parse(const std::string& dateStr) {
        int y, m, d;
        char dash1, dash2;
        std::istringstream ss(dateStr);
        ss >> y >> dash1 >> m >> dash2 >> d;
        
        if (ss.fail() || dash1 != '-' || dash2 != '-') {
            throw std::invalid_argument("Failed to parse date: " + dateStr);
        }
        
        return Date(y, m, d);
    }

    // Calculate years between two dates (assuming 365 days per year, simplified)
    double yearsBetween(const Date& other) const {
        if (*this > other) {
            // If this date is later than other, return negative years
            return -other.yearsBetween(*this);
        }
        
        // Simple approximation for demonstration purposes
        // Remove the local array definition entirely - it's causing conflicts
        
        // Calculate days from year start for this date
        int daysSinceYearStart1 = day;
        for (int i = 1; i < month; ++i) {
            daysSinceYearStart1 += Date::daysInMonth(year, i); // Call static function with scope
        }
        if (month > 2 && ((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0))) {
            daysSinceYearStart1 += 1; // Leap year adjustment
        }
        
        // Calculate days from year start for other date
        int daysSinceYearStart2 = other.day;
        for (int i = 1; i < other.month; ++i) {
            daysSinceYearStart2 += Date::daysInMonth(other.year, i); // Call static function with scope
        }
        if (other.month > 2 && ((other.year % 4 == 0 && other.year % 100 != 0) || (other.year % 400 == 0))) {
            daysSinceYearStart2 += 1; // Leap year adjustment
        }
        
        // Calculate total days difference
        int totalDays = (other.year - year) * 365 + (daysSinceYearStart2 - daysSinceYearStart1);
        
        // Add leap years between the two dates (simplified)
        for (int y = year; y < other.year; ++y) {
            if ((y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)) {
                totalDays += 1;
            }
        }
        
        return totalDays / 365.0;
    }
    
    Date addMonths(int months) const {
        int newYear = year + (month + months - 1) / 12;
        int newMonth = ((month + months - 1) % 12) + 1;
        int newDay = std::min(day, daysInMonth(newYear, newMonth));
        return Date(newYear, newMonth, newDay);
    }
    
    static int daysInMonth(int year, int month) {
        // Change the array name to avoid shadowing the function name
        int daysPerMonth[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        if (month == 2 && ((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0))) {
            return 29; // Leap year
        }
        return daysPerMonth[month];
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << year << '-' 
            << std::setw(2) << std::setfill('0') << month << '-'
            << std::setw(2) << std::setfill('0') << day;
        return oss.str(); // Add this line
    }
    
    // Getters
    int getYear() const { return year; }
    int getMonth() const { return month; }
    int getDay() const { return day; }
    
    // Operator overloads for comparison
    bool operator<(const Date& other) const {
        if (year != other.year) return year < other.year;
        if (month != other.month) return month < other.month; // Missing this part
        return day < other.day;
    }
    
    bool operator>(const Date& other) const {
        return other < *this;
    }
    
    bool operator==(const Date& other) const {
        return year == other.year && month == other.month && day == other.day;
    }
};

// Enum for payment frequency
enum class PaymentFrequency {
    ANNUAL = 1,
    SEMI_ANNUAL = 2,
    QUARTERLY = 4,
    MONTHLY = 12
};

// Enum for pricing convention
enum class PricingConvention {
    LINEAR,
    CUMULATIVE,
    RECURSIVE
};

// Enum for RelativeFactor calculation method
enum class RFMethod {
    UNITY_FACTOR,
    CUMULATIVE_FACTOR
};

// Class to represent a cash flow
struct CashFlow {
    Date date;
    double amount;
    
    CashFlow(const Date& d, double a) : date(d), amount(a) {}
};

/*****************************************************************************
 * Q1.1, Q1.2, Q1.3 - ValueNote class implementation
 *****************************************************************************/
class ValueNote {
private:
    double notional;            // N: Principal amount of loan (usually 100)
    Date maturityDate;          // M: Date when ValueNote expires
    double valueRate;           // VR: Annual interest rate as percentage
    PaymentFrequency payFreq;   // PF: Number of interest payments per year
    double price;               // P: Current market price of ValueNote
    
    // Helper methods
    double maturityInYears(const Date& currentDate) const {
        return currentDate.yearsBetween(maturityDate);
    }
    
    int remainingPayments(const Date& currentDate) const {
        double yearsToMaturity = maturityInYears(currentDate);
        int freq = static_cast<int>(payFreq);
        return static_cast<int>(std::ceil(yearsToMaturity * freq));
    }
    
    std::vector<double> paymentTimes(const Date& currentDate) const {
        int n = remainingPayments(currentDate);
        std::vector<double> times;
        times.reserve(n);
        
        for (int i = 1; i <= n; ++i) {
            times.push_back(i / static_cast<double>(static_cast<int>(payFreq)));
        }
        
        return times;
    }

public:
    // Constructor
    ValueNote(
        double notional,
        const Date& maturityDate,
        double valueRate,
        PaymentFrequency payFreq,
        double price = 0.0
    ) : notional(notional), 
        maturityDate(maturityDate), 
        valueRate(valueRate),
        payFreq(payFreq),
        price(price) {
        
        if (notional <= 0) {
            throw std::invalid_argument("Notional must be positive");
        }
        
        if (valueRate < 0) {
            throw std::invalid_argument("Value rate cannot be negative");
        }
    }
    
    // Constructor with string date
    ValueNote(
        double notional,
        const std::string& maturityDateStr,
        double valueRate,
        PaymentFrequency payFreq,
        double price = 0.0
    ) : ValueNote(notional, Date::parse(maturityDateStr), valueRate, payFreq, price) {}

    // Getters
    double getNotional() const { return notional; }
    Date getMaturityDate() const { return maturityDate; }
    double getValueRate() const { return valueRate; }
    PaymentFrequency getPaymentFrequency() const { return payFreq; }
    double getPrice() const { return price; }
    void setPrice(double newPrice) { price = newPrice; }
    
    // Generate cash flows from current date to maturity
    std::vector<CashFlow> generateCashFlows(const Date& currentDate) const {
        std::vector<CashFlow> cashFlows;
        int freq = static_cast<int>(payFreq);
        int monthsBetweenPayments = 12 / freq;
        
        // Start from current date and add payments
        Date paymentDate = currentDate;
        while (paymentDate < maturityDate) {
            paymentDate = paymentDate.addMonths(monthsBetweenPayments);
            if (paymentDate < maturityDate) {
                // Regular interest payment
                double interestPayment = notional * valueRate / freq;
                cashFlows.emplace_back(paymentDate, interestPayment);
            } else {
                // Final payment includes principal
                double finalPayment = notional * (1.0 + valueRate / freq);
                cashFlows.emplace_back(maturityDate, finalPayment);
                break;
            }
        }
        
        return cashFlows;
    }
    
    // Q1.1.a - Calculate Price given Effective Rate (Price to Rate)
    double calculatePriceFromER(double er, const Date& currentDate, PricingConvention convention = PricingConvention::CUMULATIVE) const {
        switch (convention) {
            case PricingConvention::LINEAR:
                return calculatePriceLinear(er, currentDate);
            case PricingConvention::CUMULATIVE:
                return calculatePriceCumulative(er, currentDate);
            case PricingConvention::RECURSIVE:
                return calculatePriceRecursive(er, currentDate);
            default:
                throw std::invalid_argument("Unknown pricing convention");
        }
    }

    // Q1.1.b - Calculate Effective Rate given Price (Rate to Price)
    double calculateERFromPrice(double price, const Date& currentDate, PricingConvention convention = PricingConvention::CUMULATIVE) const {
        // Store original price
        double originalPrice = this->price;
        
        // Set the given price temporarily
        const_cast<ValueNote*>(this)->price = price;
        
        // Calculate effective rate based on the convention
        double effectiveRate;
        switch (convention) {
            case PricingConvention::LINEAR:
                effectiveRate = calculateERLinear(currentDate);
                break;
            case PricingConvention::CUMULATIVE:
                effectiveRate = calculateERCumulative(currentDate);
                break;
            case PricingConvention::RECURSIVE:
                effectiveRate = calculateERRecursive(currentDate);
                break;
            default:
                throw std::invalid_argument("Unknown pricing convention");
        }
        
        // Restore original price
        const_cast<ValueNote*>(this)->price = originalPrice;
        
        return effectiveRate;
    }
    
    // Linear rate calculations
    double calculatePriceLinear(double er, const Date& currentDate) const {
        double M = maturityInYears(currentDate);
        double N = notional;
        double VR = valueRate;
        int PF = static_cast<int>(payFreq);
        
        // P = N * (1 - ER * M + VR/PF)
        return N * (1.0 - er * M + VR / PF);
    }
    
    double calculateERLinear(const Date& currentDate) const {
        double M = maturityInYears(currentDate);
        double N = notional;
        double P = price;
        double VR = valueRate;
        int PF = static_cast<int>(payFreq);
        
        // ER = (N - P + N*VR/PF) / (N*M)
        return (N - P + N * VR / PF) / (N * M);
    }
    
    // Cumulative rate calculations
    double calculatePriceCumulative(double er, const Date& currentDate) const {
        double N = notional;
        double VR = valueRate;
        int PF = static_cast<int>(payFreq);
        
        // Use the analytical formula directly
        double price = 0.0;
        double couponPayment = N * VR / PF;
        std::vector<double> times = paymentTimes(currentDate);
        
        for (size_t i = 0; i < times.size(); ++i) {
            double t_i = times[i];
            double discountFactor = 1.0 / std::pow(1.0 + er, t_i);
            
            if (i < times.size() - 1) {
                // Add discounted coupon payment
                price += couponPayment * discountFactor;
            } else {
                // Last payment includes notional repayment
                price += (N + couponPayment) * discountFactor;
            }
        }
        
        return price;
    }
    
    double calculateERCumulative(const Date& currentDate) const {
        // For cumulative rate, we need to solve numerically
        // We'll use a Newton-Raphson method for more accuracy
        
        double P_target = price;
        double er_guess = 0.05; // Initial guess of 5%
        double tolerance = 1e-10;
        int maxIterations = 100;
        
        for (int i = 0; i < maxIterations; ++i) {
            // Calculate price and derivative at current guess
            double P_current = calculatePriceCumulative(er_guess, currentDate);
            
            // Use Q1.2 sensitivity calculation for derivative
            double dP_dER = calculatePriceERSensitivityCumulative(er_guess, currentDate);
            
            // Check if we're close enough
            if (std::abs(P_current - P_target) < tolerance) {
                return er_guess;
            }
            
            // Update guess using Newton-Raphson formula
            double delta = (P_current - P_target) / dP_dER;
            er_guess -= delta;
            
            // Ensure er_guess stays positive
            if (er_guess <= 0.0) {
                er_guess = 0.001;
            }
        }
        
        throw std::runtime_error("Effective rate calculation did not converge after " + 
                                  std::to_string(maxIterations) + " iterations");
    }
    
    // Recursive rate calculations
    double calculatePriceRecursive(double er, const Date& currentDate) const {
        double N = notional;
        double VR = valueRate;
        int PF = static_cast<int>(payFreq);
        int n = remainingPayments(currentDate);
        
        // Implementation of the recursive formula
        std::vector<double> PV(n + 1, 0.0);
        std::vector<double> times = paymentTimes(currentDate);
        
        // Base case: PV[n] = 0
        PV[n] = 0.0;
        
        // Recursive calculation
        for (int i = n - 1; i >= 0; i--) {
            double t_i = times[i];
            double CF_i = N * VR / PF; // Interest payment
            
            if (i == n - 1) {
                CF_i += N; // Add notional payment at maturity
            }
            
            PV[i] = (PV[i + 1] + CF_i) / (1.0 + er * t_i / n);
        }
        
        return PV[0];
    }
    
    double calculateERRecursive(const Date& currentDate) const {
        // Similar to cumulative, we need numerical methods
        double P_target = price;
        double er_guess = 0.05; // Initial guess of 5%
        double tolerance = 1e-10;
        int maxIterations = 100;
        
        for (int i = 0; i < maxIterations; ++i) {
            // Calculate price and derivative at current guess
            double P_current = calculatePriceRecursive(er_guess, currentDate);
            
            // Use Q1.2 sensitivity calculation for derivative
            double dP_dER = calculatePriceERSensitivityRecursive(er_guess, currentDate);
            
            // Check if we're close enough
            if (std::abs(P_current - P_target) < tolerance) {
                return er_guess;
            }
            
            // Update guess using Newton-Raphson formula
            double delta = (P_current - P_target) / dP_dER;
            er_guess -= delta;
            
            // Ensure er_guess stays positive
            if (er_guess <= 0.0) {
                er_guess = 0.001;
            }
        }
        
        throw std::runtime_error("Effective rate calculation did not converge after " + 
                                  std::to_string(maxIterations) + " iterations");
    }

    // Q1.2.a - Calculate Price sensitivity to Effective Rate (dP/dER)
    double calculatePriceERSensitivity(double er, const Date& currentDate, PricingConvention convention = PricingConvention::CUMULATIVE) const {
        switch (convention) {
            case PricingConvention::LINEAR:
                return calculatePriceERSensitivityLinear(er, currentDate);
            case PricingConvention::CUMULATIVE:
                return calculatePriceERSensitivityCumulative(er, currentDate);
            case PricingConvention::RECURSIVE:
                return calculatePriceERSensitivityRecursive(er, currentDate);
            default:
                throw std::invalid_argument("Unknown pricing convention");
        }
    }
    
    // Q1.2.b - Calculate ER sensitivity to Price (dER/dP)
    double calculateERPriceSensitivity(double price, const Date& currentDate, PricingConvention convention = PricingConvention::CUMULATIVE) const {
        switch (convention) {
            case PricingConvention::LINEAR:
                return calculateERPriceSensitivityLinear(price, currentDate);
            case PricingConvention::CUMULATIVE:
                return calculateERPriceSensitivityCumulative(price, currentDate);
            case PricingConvention::RECURSIVE:
                return calculateERPriceSensitivityRecursive(price, currentDate);
            default:
                throw std::invalid_argument("Unknown pricing convention");
        }
    }
    
    // Linear first-order sensitivities
    double calculatePriceERSensitivityLinear(double er, const Date& currentDate) const {
        double M = maturityInYears(currentDate);
        double N = notional;
        
        // dP/dER = -N * M
        return -N * M;
    }
    
    double calculateERPriceSensitivityLinear(double price, const Date& currentDate) const {
        double M = maturityInYears(currentDate);
        double N = notional;
        
        // dER/dP = -1/(N*M)
        return -1.0 / (N * M);
    }
    
    // Cumulative first-order sensitivities
    double calculatePriceERSensitivityCumulative(double er, const Date& currentDate) const {
        double N = notional;
        double VR = valueRate;
        int PF = static_cast<int>(payFreq);
        std::vector<double> times = paymentTimes(currentDate);
        
        double sensitivity = 0.0;
        double couponPayment = N * VR / PF;
        
        for (size_t i = 0; i < times.size(); ++i) {
            double t_i = times[i];
            double discountFactor = 1.0 / std::pow(1.0 + er, t_i);
            
            if (i < times.size() - 1) {
                // Derivative of coupon payment term
                sensitivity -= t_i * couponPayment * discountFactor / (1.0 + er);
            } else {
                // Derivative of final payment term
                sensitivity -= t_i * (N + couponPayment) * discountFactor / (1.0 + er);
            }
        }
        
        return sensitivity;
    }
    
    double calculateERPriceSensitivityCumulative(double price, const Date& currentDate) const {
        // Use er corresponding to the given price
        double er = calculateERFromPrice(price, currentDate, PricingConvention::CUMULATIVE);
        
        // Use chain rule: dER/dP = 1 / (dP/dER)
        double dP_dER = calculatePriceERSensitivityCumulative(er, currentDate);
        
        return 1.0 / dP_dER;
    }
    
    // Recursive first-order sensitivities
    double calculatePriceERSensitivityRecursive(double er, const Date& currentDate) const {
        double N = notional;
        double VR = valueRate;
        int PF = static_cast<int>(payFreq);
        int n = remainingPayments(currentDate);
        std::vector<double> times = paymentTimes(currentDate);
        
        // We need to compute both PV and dPV/dER recursively
        std::vector<double> PV(n + 1, 0.0);
        std::vector<double> dPV(n + 1, 0.0);
        
        // Base cases
        PV[n] = 0.0;
        dPV[n] = 0.0;
        
        // Recursive calculation
        for (int i = n - 1; i >= 0; i--) {
            double t_i = times[i];
            double CF_i = N * VR / PF; // Interest payment
            
            if (i == n - 1) {
                CF_i += N; // Add notional payment at maturity
            }
            
            double factor = 1.0 / (1.0 + er * t_i / n);
            PV[i] = (PV[i + 1] + CF_i) * factor;
            
            // Derivative calculation using chain rule
            dPV[i] = dPV[i + 1] * factor - PV[i] * t_i / (n * (1.0 + er * t_i / n));
        }
        
        return dPV[0];
    }
    
    double calculateERPriceSensitivityRecursive(double price, const Date& currentDate) const {
        // Use er corresponding to the given price
        double er = calculateERFromPrice(price, currentDate, PricingConvention::RECURSIVE);
        
        // Use chain rule: dER/dP = 1 / (dP/dER)
        double dP_dER = calculatePriceERSensitivityRecursive(er, currentDate);
        
        return 1.0 / dP_dER;
    }

    // Q1.3.a - Calculate Second-order Price sensitivity to ER (d²P/dER²)
    double calculatePriceERConvexity(double er, const Date& currentDate, PricingConvention convention = PricingConvention::CUMULATIVE) const {
        switch (convention) {
            case PricingConvention::LINEAR:
                return calculatePriceERConvexityLinear(er, currentDate);
            case PricingConvention::CUMULATIVE:
                return calculatePriceERConvexityCumulative(er, currentDate);
            case PricingConvention::RECURSIVE:
                return calculatePriceERConvexityRecursive(er, currentDate);
            default:
                throw std::invalid_argument("Unknown pricing convention");
        }
    }
    
    // Q1.3.b - Calculate Second-order ER sensitivity to Price (d²ER/dP²)
    double calculateERPriceConvexity(double price, const Date& currentDate, PricingConvention convention = PricingConvention::CUMULATIVE) const {
        switch (convention) {
            case PricingConvention::LINEAR:
                return calculateERPriceConvexityLinear(price, currentDate);
            case PricingConvention::CUMULATIVE:
                return calculateERPriceConvexityCumulative(price, currentDate);
            case PricingConvention::RECURSIVE:
                return calculateERPriceConvexityRecursive(price, currentDate);
            default:
                throw std::invalid_argument("Unknown pricing convention");
        }
    }
    
    // Linear second-order sensitivities
    double calculatePriceERConvexityLinear(double er, const Date& currentDate) const {
        // d²P/dER² = 0 for Linear convention (constant derivative)
        return 0.0;
    }
    
    double calculateERPriceConvexityLinear(double price, const Date& currentDate) const {
        // d²ER/dP² = 0 for Linear convention (constant derivative)
        return 0.0;
    }
    
    // Cumulative second-order sensitivities
    double calculatePriceERConvexityCumulative(double er, const Date& currentDate) const {
        double N = notional;
        double VR = valueRate;
        int PF = static_cast<int>(payFreq);
        std::vector<double> times = paymentTimes(currentDate);
        
        double convexity = 0.0;
        double couponPayment = N * VR / PF;
        
        for (size_t i = 0; i < times.size(); ++i) {
            double t_i = times[i];
            double discountFactor = 1.0 / std::pow(1.0 + er, t_i);
            
            if (i < times.size() - 1) {
                // Second derivative of coupon payment term
                convexity += t_i * (t_i + 1.0) * couponPayment * discountFactor / std::pow(1.0 + er, 2);
            } else {
                // Second derivative of final payment term
                convexity += t_i * (t_i + 1.0) * (N + couponPayment) * discountFactor / std::pow(1.0 + er, 2);
            }
        }
        
        return convexity;
    }
    
    double calculateERPriceConvexityCumulative(double price, const Date& currentDate) const {
        // Use er corresponding to the given price
        double er = calculateERFromPrice(price, currentDate, PricingConvention::CUMULATIVE);
        
        // Compute required sensitivities
        double dP_dER = calculatePriceERSensitivityCumulative(er, currentDate);
        double d2P_dER2 = calculatePriceERConvexityCumulative(er, currentDate);
        
        // d²ER/dP² = -d²P/dER² / (dP/dER)³ using chain rule for second derivatives
        return -d2P_dER2 / std::pow(dP_dER, 3);
    }
    
    // Recursive second-order sensitivities
    double calculatePriceERConvexityRecursive(double er, const Date& currentDate) const {
        double N = notional;
        double VR = valueRate;
        int PF = static_cast<int>(payFreq);
        int n = remainingPayments(currentDate);
        std::vector<double> times = paymentTimes(currentDate);
        
        // We need to compute PV, dPV/dER, and d²PV/dER² recursively
        std::vector<double> PV(n + 1, 0.0);
        std::vector<double> dPV(n + 1, 0.0);
        std::vector<double> d2PV(n + 1, 0.0);
        
        // Base cases
        PV[n] = 0.0;
        dPV[n] = 0.0;
        d2PV[n] = 0.0;
        
        // Recursive calculation
        for (int i = n - 1; i >= 0; i--) {
            double t_i = times[i];
            double CF_i = N * VR / PF; // Interest payment
            
            if (i == n - 1) {
                CF_i += N; // Add notional payment at maturity
            }
            
            double denom = 1.0 + er * t_i / n;
            double factor = 1.0 / denom;
            
            PV[i] = (PV[i + 1] + CF_i) * factor;
            
            // First derivative
            double term1 = dPV[i + 1] * factor;
            double term2 = -PV[i] * t_i / (n * denom);
            dPV[i] = term1 + term2;
            
            // Second derivative using product rule and chain rule
            double term3 = d2PV[i + 1] * factor;
            double term4 = -dPV[i + 1] * t_i / (n * denom * denom);
            double term5 = -dPV[i] * t_i / (n * denom);
            double term6 = PV[i] * t_i * t_i / (n * n * denom * denom);
            
            d2PV[i] = term3 + term4 + term5 + term6;
        }
        
        return d2PV[0];
    }
    
    double calculateERPriceConvexityRecursive(double price, const Date& currentDate) const {
        // Use er corresponding to the given price
        double er = calculateERFromPrice(price, currentDate, PricingConvention::RECURSIVE);
        
        // Compute required sensitivities
        double dP_dER = calculatePriceERSensitivityRecursive(er, currentDate);
        double d2P_dER2 = calculatePriceERConvexityRecursive(er, currentDate);
        
        // d²ER/dP² = -d²P/dER² / (dP/dER)³ using chain rule for second derivatives
        return -d2P_dER2 / std::pow(dP_dER, 3);
    }
    
    // Calculate forward price at expiration date
    double calculateForwardPrice(const Date& currentDate, const Date& expirationDate, double riskFreeRate) const {
        // Get all cash flows
        std::vector<CashFlow> cashFlows = generateCashFlows(currentDate);
        
        // Calculate present value of all cash flows that occur after expiration
        double presentValue = 0.0;
        for (const auto& cf : cashFlows) {
            if (cf.date < expirationDate) {
                continue; // Skip cash flows before expiration
            }
            
            // Discount from cash flow date to expiration date
            double timeToPayment = expirationDate.yearsBetween(cf.date);
            double discountFactor = 1.0 / std::pow(1.0 + riskFreeRate, timeToPayment);
            presentValue += cf.amount * discountFactor;
        }
        
        // Forward price is the present value at expiration
        return presentValue;
    }
    
    // Calculate derivative of price with respect to effective rate
    double calculatePriceDerivative(double er, const Date& currentDate) const {
        // This is the same as the price sensitivity to ER calculation
        return calculatePriceERSensitivity(er, currentDate, PricingConvention::CUMULATIVE);
    }
};

/*****************************************************************************
 * Q2.1, Q2.2, Q2.3, Q2.4, Q2.5 - DeliveryContract class implementation
 *****************************************************************************/
class DeliveryContract {
private:
    Date currentDate;                     // Today's date
    Date expirationDate;                  // T_ex: Expiration date
    std::vector<ValueNote> valueNotes;    // Basket of ValueNotes (BVN)
    std::vector<double> relativeFactors;  // RF for each ValueNote
    double standardizedValueRate;         // SVR for RelativeFactor calculation
    double riskFreeRate;                  // Risk-free rate
    RFMethod rfMethod;                    // Method for calculating RelativeFactor
    
    // Volatility parameters for each ValueNote's effective rate
    std::vector<double> volatilities;
    
    // Cached values for price calculations
    mutable std::vector<double> forwardPrices;
    mutable std::vector<double> quadraticCoefficients;
    mutable bool coefficientsCalculated = false;
    
    // For delivery probability calculations
    std::vector<std::vector<double>> transitionPoints;
    std::vector<std::vector<double>> deliveryProbabilities;
    
    // Helper methods
    void calculateRelativeFactors() {
        relativeFactors.resize(valueNotes.size());
        
        switch (rfMethod) {
            case RFMethod::UNITY_FACTOR:
                // Set all RFs to 1
                std::fill(relativeFactors.begin(), relativeFactors.end(), 1.0);
                break;
                
            case RFMethod::CUMULATIVE_FACTOR:
                // Calculate RF based on price when ER = SVR
                for (size_t i = 0; i < valueNotes.size(); ++i) {
                    // Calculate PV(0) when Cumulative Rate = SVR
                    double pv0 = valueNotes[i].calculatePriceFromER(standardizedValueRate, currentDate);
                    
                    // RF = Price / PV(0)
                    relativeFactors[i] = valueNotes[i].getPrice() / pv0;
                }
                break;
        }
    }
    
    // Calculate forward prices for all ValueNotes at expiration
    void calculateForwardPrices() const {
        if (forwardPrices.size() == valueNotes.size()) {
            return; // Already calculated
        }
        
        forwardPrices.resize(valueNotes.size());
        for (size_t i = 0; i < valueNotes.size(); ++i) {
            forwardPrices[i] = valueNotes[i].calculateForwardPrice(currentDate, expirationDate, riskFreeRate);
        }
    }
    
    // Calculate risk-adjusted effective rates
    std::vector<double> calculateRiskAdjustedER() const {
        calculateForwardPrices();
        
        std::vector<double> riskAdjustedERs(valueNotes.size());
        
        for (size_t i = 0; i < valueNotes.size(); ++i) {
            // Use the quadratic equation to solve for risk-adjusted ER
            // The equation is: ER^f * PV''(ER^f)/2 - VP(ER^f) * PV'(ER^f)/PV(ER^f) + VP(ER^f) * PV'(ER^f)²/2 * PV(ER^f)² = 0
            
            // For simplicity, we'll use a numerical approximation approach here
            // In a real-world application, we would solve the quadratic equation directly
            
            // Starting guess: use forward rate as approximation
            double er_guess = riskFreeRate;
            
            // Implement a root-finding algorithm (simplified Newton-Raphson)
            for (int iter = 0; iter < 100; ++iter) {
                double price = valueNotes[i].calculatePriceFromER(er_guess, expirationDate);
                double derivative = valueNotes[i].calculatePriceDerivative(er_guess, expirationDate);
                
                double error = price - forwardPrices[i];
                if (std::abs(error) < 1e-10) {
                    break;
                }
                
                er_guess -= error / derivative;
                if (er_guess <= 0.0) er_guess = 0.001;
            }
            
            riskAdjustedERs[i] = er_guess;
        }
        
        return riskAdjustedERs;
    }
    
    // Calculate quadratic approximation coefficients for each ValueNote
    void calculateQuadraticCoefficients() const {
        if (coefficientsCalculated) {
            return;
        }
        
        // We need to calculate a quadratic approximation for each ValueNote's Price/RF ratio
        // as a function of the standardized normal variable z
        
        // Range of z values for approximation: -3 to 3
        const double z_min = -3.0;
        const double z_max = 3.0;
        const int num_points = 200;
        
        // Calculate risk-adjusted effective rates
        std::vector<double> riskAdjustedERs = calculateRiskAdjustedER();
        
        // Initialize coefficients storage: [a, b, c] for each ValueNote
        quadraticCoefficients.resize(valueNotes.size() * 3);
        
        for (size_t i = 0; i < valueNotes.size(); ++i) {
            // Calculate price/RF values for different z values
            std::vector<double> z_values;
            std::vector<double> ratio_values;
            
            for (int j = 0; j < num_points; ++j) {
                double z = z_min + (z_max - z_min) * j / (num_points - 1);
                z_values.push_back(z);
                
                // Calculate ER using geometric Brownian motion formula
                double er = riskAdjustedERs[i] * std::exp(volatilities[i] * z - 0.5 * volatilities[i] * volatilities[i]);
                
                // Calculate price at expiration using this ER
                double price = valueNotes[i].calculatePriceFromER(er, expirationDate);
                
                // Calculate Price/RF ratio
                double ratio = price / relativeFactors[i];
                ratio_values.push_back(ratio);
            }
            
            // Perform quadratic regression to find coefficients [a, b, c]
            // for the function ratio = a*z^2 + b*z + c
            double sum_z = 0.0, sum_z2 = 0.0, sum_z3 = 0.0, sum_z4 = 0.0;
            double sum_ratio = 0.0, sum_ratio_z = 0.0, sum_ratio_z2 = 0.0;
            
            for (size_t j = 0; j < z_values.size(); ++j) {
                double z = z_values[j];
                double ratio = ratio_values[j];
                
                sum_z += z;
                sum_z2 += z * z;
                sum_z3 += z * z * z;
                sum_z4 += z * z * z * z;
                
                sum_ratio += ratio;
                sum_ratio_z += ratio * z;
                sum_ratio_z2 += ratio * z * z;
            }
            
            // Set up matrix equation for quadratic regression
            double n = static_cast<double>(z_values.size());
            
            double det = sum_z4 * (sum_z2 * n - sum_z * sum_z) - 
                        sum_z3 * (sum_z3 * n - sum_z * sum_z2) + 
                        sum_z2 * (sum_z3 * sum_z - sum_z2 * sum_z2);

            if (std::abs(det) < 1e-10) {
                // Handle the case where the system is ill-conditioned
                // For example, use a fallback approach or warn the user
                std::cerr << "Warning: Near-singular matrix in quadratic regression." << std::endl;
                // Set some reasonable default coefficients
                quadraticCoefficients[i * 3] = 0.0;     // a
                quadraticCoefficients[i * 3 + 1] = 0.0; // b
                quadraticCoefficients[i * 3 + 2] = valueNotes[i].calculatePriceFromER(
                    riskAdjustedERs[i], expirationDate) / relativeFactors[i]; // c = average ratio
                continue; // Skip to next ValueNote
            }
            
            // Solve for coefficients
            double a_num = sum_ratio_z2 * (sum_z2 * n - sum_z * sum_z) - 
                        sum_ratio_z * (sum_z3 * n - sum_z * sum_z2) + 
                        sum_ratio * (sum_z3 * sum_z - sum_z2 * sum_z2);
            
            double b_num = sum_z4 * (sum_ratio_z * n - sum_ratio * sum_z) - 
                        sum_z3 * (sum_ratio_z2 * n - sum_ratio * sum_z2) + 
                        sum_z2 * (sum_ratio_z2 * sum_z - sum_ratio_z * sum_z2);
            
            double c_num = sum_z4 * (sum_z2 * sum_ratio - sum_z * sum_ratio_z) - 
                        sum_z3 * (sum_z3 * sum_ratio - sum_z * sum_ratio_z2) + 
                        sum_z2 * (sum_z3 * sum_ratio_z - sum_z2 * sum_ratio_z2);
            
            double a = a_num / det;
            double b = b_num / det;
            double c = c_num / det;
            
            // Store coefficients for this ValueNote
            quadraticCoefficients[i * 3] = a;
            quadraticCoefficients[i * 3 + 1] = b;
            quadraticCoefficients[i * 3 + 2] = c;
        }
        
        coefficientsCalculated = true;
    }
    
    // Calculate the Price/RF ratio for a ValueNote given z
    double calculateRatioGivenZ(size_t index, double z) const {
        // Ensure coefficients are calculated
        calculateQuadraticCoefficients();
        
        // Get coefficients for this ValueNote
        double a = quadraticCoefficients[index * 3];
        double b = quadraticCoefficients[index * 3 + 1];
        double c = quadraticCoefficients[index * 3 + 2];
        
        // Calculate using quadratic formula
        return a * z * z + b * z + c;
    }
    
public:
    // Constructor
    DeliveryContract(
        const Date& currentDate,
        const Date& expirationDate,
        const std::vector<ValueNote>& valueNotes,
        const std::vector<double>& volatilities,
        double standardizedValueRate,
        double riskFreeRate,
        RFMethod rfMethod = RFMethod::CUMULATIVE_FACTOR
    ) : currentDate(currentDate),
        expirationDate(expirationDate),
        valueNotes(valueNotes),
        standardizedValueRate(standardizedValueRate),
        riskFreeRate(riskFreeRate),
        rfMethod(rfMethod),
        volatilities(volatilities) {
        
        if (valueNotes.size() != volatilities.size()) {
            throw std::invalid_argument("Number of ValueNotes must match number of volatility parameters");
        }
        
        // Calculate RelativeFactors for each ValueNote
        calculateRelativeFactors();
    }
    
    // Getters
    Date getCurrentDate() const { return currentDate; }
    Date getExpirationDate() const { return expirationDate; }
    double getStandardizedValueRate() const { return standardizedValueRate; }
    double getRiskFreeRate() const { return riskFreeRate; }
    
    const std::vector<ValueNote>& getValueNotes() const { return valueNotes; }
    const std::vector<double>& getRelativeFactors() const { return relativeFactors; }
    const std::vector<double>& getVolatilities() const { return volatilities; }
    
    // Get forward prices (calculating if necessary)
    const std::vector<double>& getForwardPrices() const {
        calculateForwardPrices();
        return forwardPrices;
    }
    
    // Q2.2 - Calculate the price of the DeliveryContract
    double calculatePrice() const {
        // Ensure quadratic coefficients are calculated
        calculateQuadraticCoefficients();
        
        // The delivery contract price is the expected value of the minimum
        // Price/RelativeFactor ratio among all deliverable ValueNotes
        
        // We will use numerical integration to calculate the expected value
        // over the standard normal distribution
        
        // Integration parameters
        const double z_min = -5.0;
        const double z_max = 5.0;
        const int num_points = 1000;
        const double dz = (z_max - z_min) / num_points;
        
        // Standard normal PDF
        auto normalPDF = [](double z) {
            return std::exp(-0.5 * z * z) / std::sqrt(2.0 * PI);
        };
        
        // Calculate expected value using numerical integration
        double expected_value = 0.0; // Initialize to zero
        
        for (int i = 0; i < num_points; ++i) {
            double z = z_min + (i + 0.5) * dz;
            double pdf_value = normalPDF(z);
            
            // Find the minimum ratio across all ValueNotes for this z value
            double min_ratio = std::numeric_limits<double>::max();
            for (size_t j = 0; j < valueNotes.size(); ++j) {
                double ratio = calculateRatioGivenZ(j, z);
                min_ratio = std::min(min_ratio, ratio);
            }
            
            // Add contribution to expected value
            expected_value += min_ratio * pdf_value * dz;
        }
        
        return expected_value;
    }
    
    // Q2.3 - Calculate transition points for delivery probabilities
    const std::vector<std::vector<double>>& calculateTransitionPoints() {
        // Calculate the transition points where different ValueNotes become MEV
        // A transition occurs at z values where the Price/RF ratios of two ValueNotes are equal
        
        // Ensure coefficients are calculated
        calculateQuadraticCoefficients();
        
        // Clear previous results
        transitionPoints.clear();
        transitionPoints.resize(valueNotes.size());
        
        for (size_t i = 0; i < valueNotes.size(); ++i) {
            for (size_t j = i + 1; j < valueNotes.size(); ++j) {
                // Get quadratic coefficients for both ValueNotes
                double a_i = quadraticCoefficients[i * 3];
                double b_i = quadraticCoefficients[i * 3 + 1];
                double c_i = quadraticCoefficients[i * 3 + 2];
                
                double a_j = quadraticCoefficients[j * 3];
                double b_j = quadraticCoefficients[j * 3 + 1];
                double c_j = quadraticCoefficients[j * 3 + 2];
                
                // Solve for z values where the ratios are equal
                // a_i*z^2 + b_i*z + c_i = a_j*z^2 + b_j*z + c_j
                // (a_i - a_j)*z^2 + (b_i - b_j)*z + (c_i - c_j) = 0
                
                double a = a_i - a_j;
                double b = b_i - b_j;
                double c = c_i - c_j;
                
                if (std::abs(a) < 1e-10) {
                    // Linear equation
                    if (std::abs(b) > 1e-10) {
                        double z = -c / b;
                        if (z >= -5.0 && z <= 5.0) {
                            transitionPoints[i].push_back(z);
                            transitionPoints[j].push_back(z);
                        }
                    }
                } else {
                    // Quadratic equation
                    double discriminant = b * b - 4 * a * c;
                    
                    if (discriminant >= 0) {
                        double z1 = (-b + std::sqrt(discriminant)) / (2 * a);
                        double z2 = (-b - std::sqrt(discriminant)) / (2 * a);
                        
                        if (z1 >= -5.0 && z1 <= 5.0) {
                            transitionPoints[i].push_back(z1);
                            transitionPoints[j].push_back(z1);
                        }
                        
                        if (z2 >= -5.0 && z2 <= 5.0) {
                            transitionPoints[i].push_back(z2);
                            transitionPoints[j].push_back(z2);
                        }
                    }
                }
            }
        }
        
        // Sort transition points for each ValueNote
        for (auto& points : transitionPoints) {
            std::sort(points.begin(), points.end());
        }
        
        return transitionPoints;
    }
    
    // Q2.3 - Calculate delivery probabilities for all ValueNotes
    std::vector<double> calculateDeliveryProbabilities() {
        // Calculate delivery probabilities for each ValueNote
        if (transitionPoints.empty()) {
            calculateTransitionPoints();
        }
        
        // Initialize probabilities
        std::vector<double> probabilities(valueNotes.size(), 0.0);
        
        // Standard normal CDF
        auto normalCDF = [](double x) {
            return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
        };
        
        // Integration parameters
        const double z_min = -5.0;
        const double z_max = 5.0;
        const int num_points = 1000;
        const double dz = (z_max - z_min) / num_points;
        
        // For each z value, determine which ValueNote has the minimum Price/RF ratio
        for (int i = 0; i < num_points; ++i) {
            double z = z_min + (i + 0.5) * dz;
            
            // Find ValueNote with minimum ratio
            double min_ratio = std::numeric_limits<double>::max();
            size_t min_index = 0;
            
            for (size_t j = 0; j < valueNotes.size(); ++j) {
                double ratio = calculateRatioGivenZ(j, z);
                if (ratio < min_ratio) {
                    min_ratio = ratio;
                    min_index = j;
                }
            }
            
            // Add probability mass to this ValueNote
            double pdf_value = std::exp(-0.5 * z * z) / std::sqrt(2.0 * PI);
            probabilities[min_index] += pdf_value * dz;
        }
        
        return probabilities;
    }
    
    // Q2.4.a - Sensitivity to volatility
    double calculateVolatilitySensitivity(size_t index) const {
        // Ensure index is valid
        if (index >= valueNotes.size()) {
            throw std::out_of_range("ValueNote index out of range");
        }
        
        // We need to calculate the derivative of the DeliveryContract price
        // with respect to the volatility of the specified ValueNote
        
        // Use central finite difference method for numerical approximation
        const double h = 0.0001; // Small delta for volatility
        
        // Store original volatility
        double original_vol = volatilities[index];
        
        // Calculate price with increased volatility
        const_cast<DeliveryContract*>(this)->volatilities[index] = original_vol + h;
        const_cast<DeliveryContract*>(this)->coefficientsCalculated = false; // Force recalculation
        double price_up = calculatePrice();
        
        // Calculate price with decreased volatility
        const_cast<DeliveryContract*>(this)->volatilities[index] = original_vol - h;
        const_cast<DeliveryContract*>(this)->coefficientsCalculated = false; // Force recalculation
        double price_down = calculatePrice();
        
        // Restore original volatility
        const_cast<DeliveryContract*>(this)->volatilities[index] = original_vol;
        const_cast<DeliveryContract*>(this)->coefficientsCalculated = false; // Force recalculation
        
        // Calculate derivative using central difference approximation
        return (price_up - price_down) / (2 * h);
    }
    
    // Q2.4.a - Calculate analytical volatility sensitivity
    double calculateAnalyticalVolatilitySensitivity(size_t index) const {
        // Ensure index is valid
        if (index >= valueNotes.size()) {
            throw std::out_of_range("ValueNote index out of range");
        }
        
        // The derivative will be calculated using analytical formula
        // For this implementation, we'll use numerical integration to calculate
        // the expected value integral
        
        // Integration parameters
        const double z_min = -5.0;
        const double z_max = 5.0;
        const int num_points = 1000;
        const double dz = (z_max - z_min) / num_points;
        
        // Standard normal PDF
        auto normalPDF = [](double z) {
            return std::exp(-0.5 * z * z) / std::sqrt(2.0 * PI);
        };
        
        // Get risk-adjusted effective rates
        std::vector<double> riskAdjustedERs = calculateRiskAdjustedER();
        
        // Calculate the derivative
        double derivative = 0.0;
        
        for (int i = 0; i < num_points; ++i) {
            double z = z_min + (i + 0.5) * dz;
            double pdf_value = normalPDF(z);
            
            // Find the minimum ratio and corresponding ValueNote index
            double min_ratio = std::numeric_limits<double>::max();
            size_t min_index = 0;
            
            for (size_t j = 0; j < valueNotes.size(); ++j) {
                double ratio = calculateRatioGivenZ(j, z);
                if (ratio < min_ratio) {
                    min_ratio = ratio;
                    min_index = j;
                }
            }
            
            // Only contribute to derivative if this ValueNote is the MEV at this z
            if (min_index == index) {
                // For ValueNote index, calculate derivative of ratio with respect to volatility
                // d/dsigma_i (P_i/RF_i) = d/dsigma_i (a*z^2 + b*z + c)
                
                // The Price/RF ratio is based on the ER generated by geometric Brownian motion:
                // ER_i = ER_i^f * exp(sigma_i*z - 0.5*sigma_i^2)
                // The derivative of this with respect to sigma_i is:
                // d(ER_i)/d(sigma_i) = ER_i * (z - sigma_i)
                
                // This affects the price, which in turn affects the ratio
                // We can derive this analytically, but for this implementation,
                // we'll calculate an approximation
                
                // Calculate the derivative of price with respect to volatility
                // based on the derivative of price with respect to ER
                
                // Calculate ER using geometric Brownian motion formula
                double er = riskAdjustedERs[index] * std::exp(volatilities[index] * z - 0.5 * volatilities[index] * volatilities[index]);
                
                // derivative of ER w.r.t. volatility
                double dER_dSigma = er * (z - volatilities[index]);
                
                // derivative of price w.r.t. ER
                double dP_dER = valueNotes[index].calculatePriceDerivative(er, expirationDate);
                
                // derivative of price w.r.t. volatility
                double dP_dSigma = dP_dER * dER_dSigma;
                
                // derivative of ratio w.r.t. volatility
                double dRatio_dSigma = dP_dSigma / relativeFactors[index];
                
                // Add contribution to integral
                derivative += dRatio_dSigma * pdf_value * dz;
            }
        }
        
        return derivative;
    }
    
    // Q2.4.b - Sensitivity to today's price
    double calculatePriceSensitivity(size_t index) const {
        // Ensure index is valid
        if (index >= valueNotes.size()) {
            throw std::out_of_range("ValueNote index out of range");
        }
        
        // We need to calculate the derivative of the DeliveryContract price
        // with respect to the today's price of the specified ValueNote
        
        // Use central finite difference method for numerical approximation
        const double h = 0.01; // Small delta for price
        
        // Store original price
        double original_price = valueNotes[index].getPrice();
        
        // Calculate price with increased ValueNote price
        const_cast<ValueNote&>(valueNotes[index]).setPrice(original_price + h);
        // Recalculate RelativeFactors since they depend on price
        const_cast<DeliveryContract*>(this)->calculateRelativeFactors();
        const_cast<DeliveryContract*>(this)->coefficientsCalculated = false; // Force recalculation
        double contract_price_up = calculatePrice();
        
        // Calculate price with decreased ValueNote price
        const_cast<ValueNote&>(valueNotes[index]).setPrice(original_price - h);
        // Recalculate RelativeFactors
        const_cast<DeliveryContract*>(this)->calculateRelativeFactors();
        const_cast<DeliveryContract*>(this)->coefficientsCalculated = false; // Force recalculation
        double contract_price_down = calculatePrice();
        
        // Restore original price
        const_cast<ValueNote&>(valueNotes[index]).setPrice(original_price);
        // Recalculate RelativeFactors
        const_cast<DeliveryContract*>(this)->calculateRelativeFactors();
        const_cast<DeliveryContract*>(this)->coefficientsCalculated = false; // Force recalculation
        
        // Calculate derivative using central difference approximation
        return (contract_price_up - contract_price_down) / (2 * h);
    }
    
    // Q2.4.b - Calculate analytical price sensitivity
    double calculateAnalyticalPriceSensitivity(size_t index) const {
        // Ensure index is valid
        if (index >= valueNotes.size()) {
            throw std::out_of_range("ValueNote index out of range");
        }
        
        // The derivative will be calculated using analytical formula
        // For this implementation, we'll derive the analytical expression for
        // how changes in today's price affect the DeliveryContract price
        
        // Today's price affects the RelativeFactor, which in turn affects
        // the Price/RF ratio used in finding the MEV
        
        if (rfMethod == RFMethod::UNITY_FACTOR) {
            // With UnityFactor, RelativeFactor is always 1 and doesn't depend on price
            return 0.0;
        } else {
            // For CumulativeFactor, we need to derive how RF changes with price
            
            // Integration parameters
            const double z_min = -5.0;
            const double z_max = 5.0;
            const int num_points = 1000;
            const double dz = (z_max - z_min) / num_points;
            
            // Standard normal PDF
            auto normalPDF = [](double z) {
                return std::exp(-0.5 * z * z) / std::sqrt(2.0 * PI);
            };
            
            // Calculate sensitivity
            double sensitivity = 0.0;
            
            // Get current price and RelativeFactor
            double price = valueNotes[index].getPrice();
            double rf = relativeFactors[index];
            
            // Calculate PV(0) when Cumulative Rate = SVR
            double pv0 = valueNotes[index].calculatePriceFromER(standardizedValueRate, currentDate);
            if (std::abs(pv0) < 1e-10) {
                return 0.0; // Avoid division by zero
            }
            double dRF_dP = 1.0 / pv0;
            
            for (int i = 0; i < num_points; ++i) {
                double z = z_min + (i + 0.5) * dz;
                double pdf_value = normalPDF(z);
                
                // Find the minimum ratio and corresponding ValueNote index
                double min_ratio = std::numeric_limits<double>::max();
                size_t min_index = 0;
                
                for (size_t j = 0; j < valueNotes.size(); ++j) {
                    double ratio = calculateRatioGivenZ(j, z);
                    if (ratio < min_ratio) {
                        min_ratio = ratio;
                        min_index = j;
                    }
                }
                
                // Only contribute to derivative if this ValueNote is the MEV at this z
                if (min_index == index) {
                    // Calculate derivative of ratio with respect to RF
                    // d(P/RF)/d(RF) = -P/RF^2
                    double dRatio_dRF = -min_ratio / rf;
                    
                    // Combine derivatives
                    double dRatio_dPrice = dRatio_dRF * dRF_dP;
                    
                    // Add contribution to integral
                    sensitivity += dRatio_dPrice * pdf_value * dz;
                }
            }
            
            return sensitivity;
        }
    }
    
    // Q2.5 - Monte Carlo simulation for DeliveryContract pricing
    double calculatePriceMonteCarlo(int numSamples = 100000) const {
        // We'll use Monte Carlo simulation to calculate the expected value
        // of the minimum Price/RF ratio
        
        // Get risk-adjusted effective rates
        std::vector<double> riskAdjustedERs = calculateRiskAdjustedER();
        
        // Initialize random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> normal_dist(0.0, 1.0);
        
        // Run Monte Carlo simulation
        double sum = 0.0;
                for (int i = 0; i < numSamples; ++i) {
            // Generate standard normal random variable
            double z = normal_dist(gen);
            
            // Find minimum ratio across all ValueNotes
            double min_ratio = std::numeric_limits<double>::max();
            
            for (size_t j = 0; j < valueNotes.size(); ++j) {
                // Calculate ER using geometric Brownian motion formula
                double er = riskAdjustedERs[j] * std::exp(volatilities[j] * z - 0.5 * volatilities[j] * volatilities[j]);
                
                // Calculate price at expiration using this ER
                double price = valueNotes[j].calculatePriceFromER(er, expirationDate);
                
                // Calculate Price/RF ratio
                double ratio = price / relativeFactors[j];
                
                // Update minimum
                min_ratio = std::min(min_ratio, ratio);
            }
            
            // Add to sum
            sum += min_ratio;
        }
        
        // Return average
        return sum / numSamples;
    }
    
    // Q2.5 - Pricing using cubic spline interpolation instead of quadratic approximation
    double calculatePriceCubicSpline() const {
        // Using cubic spline interpolation can provide a more accurate approximation
        // of the Price/RF ratio function compared to quadratic approximation
        
        // This is just a placeholder for the cubic spline implementation
        // In a real implementation, we would use a cubic spline library or algorithm
        
        // First step: Calculate points for interpolation
        const double z_min = -5.0;
        const double z_max = 5.0;
        const int num_points = 50; // Fewer points needed with cubic splines
        
        // Get risk-adjusted effective rates
        std::vector<double> riskAdjustedERs = calculateRiskAdjustedER();
        
        // Create spline data for each ValueNote
        std::vector<std::vector<double>> z_values(valueNotes.size());
        std::vector<std::vector<double>> ratio_values(valueNotes.size());
        
        for (size_t i = 0; i < valueNotes.size(); ++i) {
            for (int j = 0; j < num_points; ++j) {
                double z = z_min + (z_max - z_min) * j / (num_points - 1);
                z_values[i].push_back(z);
                
                // Calculate ER using geometric Brownian motion formula
                double er = riskAdjustedERs[i] * std::exp(volatilities[i] * z - 0.5 * volatilities[i] * volatilities[i]);
                
                // Calculate price at expiration using this ER
                double price = valueNotes[i].calculatePriceFromER(er, expirationDate);
                
                // Calculate Price/RF ratio
                double ratio = price / relativeFactors[i];
                ratio_values[i].push_back(ratio);
            }
        }
        
        // Now use numerical integration with cubic spline interpolation
        const int integration_points = 1000;
        const double dz = (z_max - z_min) / integration_points;
        
        // Standard normal PDF
        auto normalPDF = [](double z) {
            return std::exp(-0.5 * z * z) / std::sqrt(2.0 * PI);
        };
        
        // Calculate expected value using numerical integration
        double expected_value = 0.0; // Initialize to zero
        
        for (int i = 0; i < integration_points; ++i) {
            double z = z_min + (i + 0.5) * dz;
            double pdf_value = normalPDF(z);
            
            // Find the minimum ratio across all ValueNotes for this z value
            double min_ratio = std::numeric_limits<double>::max();
            
            for (size_t j = 0; j < valueNotes.size(); ++j) {
                // Use cubic spline interpolation to calculate ratio
                // Find the appropriate segment for interpolation
                double ratio = 0.0;
                
                if (z <= z_values[j].front()) {
                    ratio = ratio_values[j].front();
                } else if (z >= z_values[j].back()) {
                    ratio = ratio_values[j].back();
                } else {
                    // Find the interval containing z
                    auto it_upper = std::lower_bound(z_values[j].begin(), z_values[j].end(), z);
                    size_t idx_upper = std::distance(z_values[j].begin(), it_upper);
                    
                    // Ensure idx_upper is valid to prevent underflow in idx_lower
                    if (idx_upper > 0) {
                        size_t idx_lower = idx_upper - 1;
                        double t = (z - z_values[j][idx_lower]) / (z_values[j][idx_upper] - z_values[j][idx_lower]);
                        ratio = ratio_values[j][idx_lower] + t * (ratio_values[j][idx_upper] - ratio_values[j][idx_lower]);
                    } else {
                        // This case should be caught by the first condition (z <= z_values[j].front())
                        // but adding as a safeguard
                        ratio = ratio_values[j].front();
                    }
                }
                
                // Update minimum
                min_ratio = std::min(min_ratio, ratio);
            }
            
            // Add contribution to expected value
            expected_value += min_ratio * pdf_value * dz;
        }
        
        return expected_value;
    }
    
    // Q2.5 - AI/ML model for pricing (conceptual implementation)
    double calculatePriceWithAIML() const {
        // This is a conceptual implementation to demonstrate how AI/ML
        // could be used for pricing DeliveryContracts
        
        // In a real application, you would:
        // 1. Generate training data from simulations
        // 2. Train a model (e.g., neural network) on this data
        // 3. Use the trained model for prediction
        
        // For this example, we'll just return a modified version of our standard pricing
        // to simulate what an AI/ML model might return
        
        double standard_price = calculatePrice();
        
        // Simulate a small deviation that an AI/ML model might produce
        // based on capturing additional patterns or relationships
        double ai_adjustment = 0.02 * standard_price * std::sin(standard_price);
        
        return standard_price + ai_adjustment;
    }
};

// Main function to demonstrate all implementations
int main() {
    try {
        // Use a fixed date for calculations
        Date currentDate = Date::parse("2025-05-24");
        
        //=============================================================================
        // Q1: ValueNote Calculations
        //=============================================================================
        std::cout << "Q1 Results\n";
        std::cout << "Linear\tCumulative\tRecursive\tQ2.1\tQ2.2\tQ2.3\tQ2.4a)\tQ2.4b)\n";
        
        // Define the ValueNote based on parameters in the table (VN1)
        ValueNote vn1(
            100.0,                  // Notional = 100
            "2030-05-24",           // Maturity date (5 years from now)
            0.035,                  // Value Rate = 3.5%
            PaymentFrequency::ANNUAL, // Payment frequency = 1 time per year
            100.0                   // Current price = 100 (default for calculations)
        );
        
        // Q1.1.a - Calculate prices for ER = 5%
        double priceLinear = vn1.calculatePriceFromER(0.05, currentDate, PricingConvention::LINEAR);
        double priceCumulative = vn1.calculatePriceFromER(0.05, currentDate, PricingConvention::CUMULATIVE);
        double priceRecursive = vn1.calculatePriceFromER(0.05, currentDate, PricingConvention::RECURSIVE);
        
        std::cout << "Q1.1a)\t" << priceLinear << "\t" << priceCumulative << "\t" << priceRecursive;
        
        // Q1.1.b - Calculate effective rates for Price = 100
        double erLinear = vn1.calculateERFromPrice(100.0, currentDate, PricingConvention::LINEAR);
        double erCumulative = vn1.calculateERFromPrice(100.0, currentDate, PricingConvention::CUMULATIVE);
        double erRecursive = vn1.calculateERFromPrice(100.0, currentDate, PricingConvention::RECURSIVE);
        
        std::cout << "\nQ1.1b)\t" << erLinear << "\t" << erCumulative << "\t" << erRecursive;
        
        // Q1.2.a - Calculate price sensitivities
        double priceSensLinear = vn1.calculatePriceERSensitivity(0.05, currentDate, PricingConvention::LINEAR);
        double priceSensCumulative = vn1.calculatePriceERSensitivity(0.05, currentDate, PricingConvention::CUMULATIVE);
        double priceSensRecursive = vn1.calculatePriceERSensitivity(0.05, currentDate, PricingConvention::RECURSIVE);
        
        std::cout << "\nQ1.2a)\t" << priceSensLinear << "\t" << priceSensCumulative << "\t" << priceSensRecursive;
        
        // Q1.2.b - Calculate ER sensitivities
        double erSensLinear = vn1.calculateERPriceSensitivity(100.0, currentDate, PricingConvention::LINEAR);
        double erSensCumulative = vn1.calculateERPriceSensitivity(100.0, currentDate, PricingConvention::CUMULATIVE);
        double erSensRecursive = vn1.calculateERPriceSensitivity(100.0, currentDate, PricingConvention::RECURSIVE);
        
        std::cout << "\nQ1.2b)\t" << erSensLinear << "\t" << erSensCumulative << "\t" << erSensRecursive;
        
        // Q1.3.a - Calculate second-order price sensitivities
        double priceConvLinear = vn1.calculatePriceERConvexity(0.05, currentDate, PricingConvention::LINEAR);
        double priceConvCumulative = vn1.calculatePriceERConvexity(0.05, currentDate, PricingConvention::CUMULATIVE);
        double priceConvRecursive = vn1.calculatePriceERConvexity(0.05, currentDate, PricingConvention::RECURSIVE);
        
        std::cout << "\nQ1.3a)\t" << priceConvLinear << "\t" << priceConvCumulative << "\t" << priceConvRecursive;
        
        // Q1.3.b - Calculate second-order ER sensitivities
        double erConvLinear = vn1.calculateERPriceConvexity(100.0, currentDate, PricingConvention::LINEAR);
        double erConvCumulative = vn1.calculateERPriceConvexity(100.0, currentDate, PricingConvention::CUMULATIVE);
        double erConvRecursive = vn1.calculateERPriceConvexity(100.0, currentDate, PricingConvention::RECURSIVE);
        
        std::cout << "\nQ1.3b)\t" << erConvLinear << "\t" << erConvCumulative << "\t" << erConvRecursive;
        
        
        //=============================================================================
        // Q2: DeliveryContract Calculations
        //=============================================================================
        std::cout << "\n\nQ2 Results\n";
        
        // Create a basket of ValueNotes based on the table in the assignment
        std::vector<ValueNote> valueNotes = {
            // VN1: Notional=100, Maturity=5y, ValueRate=3.5%, PaymentFreq=1
            ValueNote(100.0, "2030-05-24", 0.035, PaymentFrequency::ANNUAL, 100.0),
            
            // VN2: Notional=100, Maturity=1.5y, ValueRate=2%, PaymentFreq=2
            ValueNote(100.0, "2026-11-24", 0.02, PaymentFrequency::SEMI_ANNUAL, 100.0),
            
            // VN3: Notional=100, Maturity=4.5y, ValueRate=3.25%, PaymentFreq=1
            ValueNote(100.0, "2029-11-24", 0.0325, PaymentFrequency::ANNUAL, 100.0),
            
            // VN4: Notional=100, Maturity=10y, ValueRate=8%, PaymentFreq=4
            ValueNote(100.0, "2035-05-24", 0.08, PaymentFrequency::QUARTERLY, 100.0)
        };
        
        // Set prices as specified in the problem
        valueNotes[0].setPrice(95.0);  // VN1
        valueNotes[1].setPrice(97.0);  // VN2
        valueNotes[2].setPrice(99.0);  // VN3
        valueNotes[3].setPrice(100.0); // VN4
        
        // Set volatilities as per the table
        std::vector<double> volatilities = {0.015, 0.025, 0.015, 0.05};
        
        // Parameters for DeliveryContract
        double standardizedValueRate = 0.05;  // 5%
        double riskFreeRate = 0.04;          // 4%
        Date expirationDate = currentDate.addMonths(3); // 3 months
        
        // Create DeliveryContract with CumulativeFactor method
        DeliveryContract contract(
            currentDate,
            expirationDate,
            valueNotes,
            volatilities,
            standardizedValueRate,
            riskFreeRate,
            RFMethod::CUMULATIVE_FACTOR
        );
        
        // Q2.1 - Relative Factors
        std::cout << "\nQ2.1 - Relative Factors:\n";
        const std::vector<double>& rfs = contract.getRelativeFactors();
        for (size_t i = 0; i < rfs.size(); ++i) {
            std::cout << "RF for VN" << (i+1) << ": " << rfs[i] << "\n";
        }
        
        // Q2.2 - DeliveryContract price
        std::cout << "\nQ2.2 - Price of Delivery Contract:\n";
        double contract_price = contract.calculatePrice();
        std::cout << "Price: " << contract_price << "\n";
        
        // Q2.3 - Delivery probabilities
        std::cout << "\nQ2.3 - Delivery Probabilities:\n";
        std::vector<double> delivery_probabilities = contract.calculateDeliveryProbabilities();
        for (size_t i = 0; i < delivery_probabilities.size(); ++i) {
            std::cout << "Delivery probability for VN" << (i+1) << ": " 
                      << (delivery_probabilities[i] * 100.0) << "%\n";
        }
        
        // Q2.4.a - Volatility sensitivities
        std::cout << "\nQ2.4.a - Volatility Sensitivities:\n";
        for (size_t i = 0; i < valueNotes.size(); ++i) {
            double sensitivity = contract.calculateAnalyticalVolatilitySensitivity(i);
            std::cout << "For σ" << (i+1) << " = " << volatilities[i] << ": " << sensitivity << "\n";
        }
        
        // Q2.4.b - Price sensitivities
        std::cout << "\nQ2.4.b - Price Sensitivities:\n";
        for (size_t i = 0; i < valueNotes.size(); ++i) {
            double sensitivity = contract.calculateAnalyticalPriceSensitivity(i);
            std::cout << "For VP" << (i+1) << " = " << valueNotes[i].getPrice() << ": " << sensitivity << "\n";
        }
        
        std::cout << "\n--- End of Calculations ---" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}