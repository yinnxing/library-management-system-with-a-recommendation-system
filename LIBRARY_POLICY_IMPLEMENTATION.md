# Library Management System - Policy Implementation

## Overview
This document describes the implementation of library policies in the Library Management System, including borrowing limits, overdue fees, and reservation management.

## Library Policies Implemented

### 1. Maximum Borrowing Limit
- **Limit**: 5 books per user
- **Scope**: Maximum number of books borrowed at the same time
- **Status**: BORROWED transactions count towards this limit

### 2. Maximum Reservation Limit  
- **Limit**: 3 books per user
- **Scope**: Maximum number of books reserved for simultaneous borrowing
- **Status**: PENDING transactions count towards this limit

### 3. Maximum Borrowing Period
- **Duration**: 14 days
- **Calculation**: From borrowing date (when status changes to BORROWED) to expected return date
- **Implementation**: Due date is set when transaction status changes from PENDING to BORROWED

### 4. Overdue Fee
- **Rate**: 5,000 VND per day
- **Calculation**: From the day after due_date
- **Implementation**: Calculated automatically by scheduled tasks

### 5. Holding Period
- **Duration**: 3 days
- **Scope**: From the time the book is ready (PENDING status) to the time the user must come to pick it up
- **Implementation**: Pickup deadline is set when reservation is created

## Technical Implementation

### 1. Database Changes

#### Transaction Model Updates
```java
// New fields added to Transaction entity
private LocalDateTime pickupDeadline;  // 3 days from borrow_date for PENDING status
private BigDecimal overdueFee;         // Calculated overdue fee in VND
```

#### TransactionStatus Enum
```java
public enum TransactionStatus {
    PENDING,   // Waiting to pick up book (not yet received at library)
    BORROWED,  // Book has been borrowed (received at library)
    RETURNED,  // Book has been returned
    CANCELLED, // Transaction cancelled (when overdue and book not picked up)
    OVERDUE    // Overdue borrowed book
}
```

### 2. New Services

#### LibraryPolicyService
Central service for all policy validations and enforcement:

- **Policy Constants**: Centralized configuration of all policy limits
- **Validation Methods**: 
  - `validateBorrowingEligibility()` - Checks borrowing limits and user status
  - `validateReservationEligibility()` - Checks reservation limits and user status
- **Fee Calculation**: Automatic overdue fee calculation
- **Scheduled Tasks**: 
  - Daily cancellation of expired pending transactions
  - Daily update of overdue transactions and fees

#### Enhanced TransactionService
Updated with policy integration:

- **Policy Validation**: All borrowing/reservation operations validate against policies
- **Fee Management**: Methods for paying overdue fees
- **Statistics**: User borrowing statistics including current limits

### 3. New API Endpoints

#### TransactionController Enhancements

```http
GET /api/transactions/stats/{userId}
```
Returns user's borrowing statistics including current borrowed books, reservations, overdue books, and unpaid fees.

```http
POST /api/transactions/pay-fees/{userId}?amount={amount}
```
Allows users to pay overdue fees.

```http
PUT /api/transactions/{transactionId}/cancel
```
Cancels a pending transaction.

```http
GET /api/transactions/policy
```
Returns library policy information including all limits and fees.

### 4. Error Handling

New error codes for policy violations:

- `BORROWING_LIMIT_EXCEEDED` (2001): Maximum 5 books borrowing limit exceeded
- `RESERVATION_LIMIT_EXCEEDED` (2002): Maximum 3 books reservation limit exceeded  
- `PICKUP_DEADLINE_EXPIRED` (2003): 3-day pickup deadline expired
- `OVERDUE_BOOKS_EXIST` (2004): User has overdue books
- `UNPAID_OVERDUE_FEES` (2005): User has unpaid overdue fees

### 5. Automated Tasks

#### Scheduled Tasks (Daily at Midnight)

1. **Cancel Expired Pending Transactions**
   - Finds PENDING transactions past pickup deadline (3 days)
   - Changes status to CANCELLED
   - Returns book to available inventory

2. **Update Overdue Transactions**
   - Finds BORROWED transactions past due date
   - Changes status to OVERDUE
   - Calculates and updates overdue fees

## Policy Enforcement Flow

### Borrowing a Book (Creating Reservation)
1. Check reservation limit (max 3 pending)
2. Check for existing overdue books
3. Check for unpaid overdue fees
4. Create PENDING transaction with pickup deadline (3 days)
5. Reduce available book quantity

### Picking Up Reserved Book (PENDING → BORROWED)
1. Check borrowing limit (max 5 borrowed)
2. Check for existing overdue books
3. Check for unpaid overdue fees
4. Set due date (14 days from pickup)
5. Change status to BORROWED

### Returning a Book
1. Calculate final overdue fee if applicable
2. Change status to RETURNED
3. Set return date
4. Increase available book quantity

## Usage Examples

### Check User Borrowing Stats
```http
GET /api/transactions/stats/user123
```

Response:
```json
{
  "code": 1000,
  "message": "Thống kê mượn sách của người dùng",
  "result": {
    "currentBorrowedBooks": 3,
    "currentReservations": 1,
    "overdueBooks": 0,
    "unpaidOverdueFees": 0,
    "maxBorrowingLimit": 5,
    "maxReservationLimit": 3
  }
}
```

### Pay Overdue Fees
```http
POST /api/transactions/pay-fees/user123?amount=15000
```

### Get Library Policy
```http
GET /api/transactions/policy
```

Response:
```json
{
  "code": 1000,
  "message": "Thông tin chính sách thư viện",
  "result": {
    "maxBorrowingLimit": 5,
    "maxReservationLimit": 3,
    "borrowingPeriodDays": 14,
    "holdingPeriodDays": 3,
    "overdueFeePerDay": 5000
  }
}
```

## Benefits

1. **Automated Policy Enforcement**: All library policies are automatically enforced
2. **Fair Resource Distribution**: Limits ensure books are available to all users
3. **Revenue Generation**: Overdue fees encourage timely returns
4. **Inventory Management**: Automatic cancellation of expired reservations
5. **User Transparency**: Clear policy information and borrowing statistics
6. **Scalable Architecture**: Centralized policy service for easy maintenance

## Configuration

All policy constants are centralized in `LibraryPolicyService`:

```java
public static final int MAX_BORROWING_LIMIT = 5;
public static final int MAX_RESERVATION_LIMIT = 3;
public static final int BORROWING_PERIOD_DAYS = 14;
public static final int HOLDING_PERIOD_DAYS = 3;
public static final BigDecimal OVERDUE_FEE_PER_DAY = new BigDecimal("5000");
```

These can be easily modified or moved to configuration files for dynamic updates. 