package com.library.booksystem.enums;

public enum TransactionStatus {
    PENDING,   // Chờ mượn sách (chưa nhận sách tại thư viện)
    BORROWED,  // Đã mượn sách (sách đã được nhận tại thư viện)
    RETURNED,  // Sách đã được trả lại
    CANCELLED, // Giao dịch bị hủy (khi quá hạn mà không nhận sách)
    OVERDUE    // Quá hạn mượn sách
}
