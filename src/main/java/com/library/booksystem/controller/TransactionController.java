package com.library.booksystem.controller;

import com.library.booksystem.dto.request.TransactionRequest;
import com.library.booksystem.dto.response.ApiResponse;
import com.library.booksystem.dto.response.TransactionResponse;
import com.library.booksystem.service.LibraryPolicyService;
import com.library.booksystem.service.TransactionService;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.apache.coyote.BadRequestException;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.List;

@RestController
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
@RequiredArgsConstructor
@RequestMapping("/transactions")
public class TransactionController {
    TransactionService transactionService;
    
    @PostMapping("/search")
    public List<TransactionResponse> getTransactions(@RequestBody TransactionRequest transactionRequest) {
        return transactionService.getTransactions(transactionRequest);
    }
    
    /**
     * Get user's borrowing statistics including limits and current status
     */
    @GetMapping("/stats/{userId}")
    public ApiResponse<LibraryPolicyService.BorrowingStats> getUserBorrowingStats(@PathVariable String userId) {
        return ApiResponse.<LibraryPolicyService.BorrowingStats>builder()
                .result(transactionService.getUserBorrowingStats(userId))
                .message("Thống kê mượn sách của người dùng")
                .build();
    }
    
    /**
     * Pay overdue fees for a user
     */
    @PostMapping("/pay-fees/{userId}")
    public ApiResponse<Void> payOverdueFees(@PathVariable String userId, @RequestParam BigDecimal amount) {
        transactionService.payOverdueFees(userId, amount);
        return ApiResponse.<Void>builder()
                .message("Thanh toán phí trễ hạn thành công")
                .build();
    }
    
    /**
     * Cancel a pending transaction
     */
    @PutMapping("/{transactionId}/cancel")
    public ApiResponse<TransactionResponse> cancelPendingTransaction(@PathVariable String transactionId) throws BadRequestException {
        TransactionResponse cancelledTransaction = transactionService.cancelPendingTransaction(transactionId);
        return ApiResponse.<TransactionResponse>builder()
                .result(cancelledTransaction)
                .message("Giao dịch đã được hủy thành công")
                .build();
    }
    
    /**
     * Get library policy information
     */
    @GetMapping("/policy")
    public ApiResponse<PolicyInfo> getLibraryPolicy() {
        PolicyInfo policyInfo = PolicyInfo.builder()
                .maxBorrowingLimit(LibraryPolicyService.MAX_BORROWING_LIMIT)
                .maxReservationLimit(LibraryPolicyService.MAX_RESERVATION_LIMIT)
                .borrowingPeriodDays(LibraryPolicyService.BORROWING_PERIOD_DAYS)
                .holdingPeriodDays(LibraryPolicyService.HOLDING_PERIOD_DAYS)
                .overdueFeePerDay(LibraryPolicyService.OVERDUE_FEE_PER_DAY)
                .build();
        
        return ApiResponse.<PolicyInfo>builder()
                .result(policyInfo)
                .message("Thông tin chính sách thư viện")
                .build();
    }
    
    /**
     * Inner class for policy information
     */
    @lombok.Data
    @lombok.Builder
    public static class PolicyInfo {
        private int maxBorrowingLimit;
        private int maxReservationLimit;
        private int borrowingPeriodDays;
        private int holdingPeriodDays;
        private BigDecimal overdueFeePerDay;
    }
}
