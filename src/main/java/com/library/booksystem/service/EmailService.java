package com.library.booksystem.service;

import com.library.booksystem.model.Transaction;
import com.library.booksystem.model.User;
import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import java.time.format.DateTimeFormatter;

@Service
@FieldDefaults(level = AccessLevel.PRIVATE)
@Slf4j
public class EmailService {
    
    final JavaMailSender mailSender;
    
    @Value("${library.email.from}")
    String fromEmail;
    
    @Value("${library.email.name}")
    String libraryName;
    
    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm");
    
    public EmailService(JavaMailSender mailSender) {
        this.mailSender = mailSender;
    }
    
    /**
     * Send email notification when a book is successfully borrowed
     */
    @Async("emailTaskExecutor")
    public void sendBookBorrowedNotification(Transaction transaction) {
        try {
            User user = transaction.getUser();
            String subject = "Book Borrowed Successfully - " + transaction.getBook().getTitle();
            
            String htmlContent = buildBookBorrowedEmailContent(transaction);
            
            sendHtmlEmail(user.getEmail(), subject, htmlContent);
            log.info("Book borrowed notification sent to user: {}", user.getEmail());
            
        } catch (Exception e) {
            log.error("Failed to send book borrowed notification to user: {}", 
                    transaction.getUser().getEmail(), e);
        }
    }
    
    /**
     * Send reminder email 3 days before due date
     */
    @Async("emailTaskExecutor")
    public void sendReturnReminderNotification(Transaction transaction) {
        try {
            User user = transaction.getUser();
            String subject = "Return Reminder - " + transaction.getBook().getTitle();
            
            String htmlContent = buildReturnReminderEmailContent(transaction);
            
            sendHtmlEmail(user.getEmail(), subject, htmlContent);
            log.info("Return reminder notification sent to user: {}", user.getEmail());
            
        } catch (Exception e) {
            log.error("Failed to send return reminder notification to user: {}", 
                    transaction.getUser().getEmail(), e);
        }
    }
    
    /**
     * Send overdue notification
     */
    @Async("emailTaskExecutor")
    public void sendOverdueNotification(Transaction transaction) {
        try {
            User user = transaction.getUser();
            String subject = "OVERDUE NOTICE - " + transaction.getBook().getTitle();
            
            String htmlContent = buildOverdueEmailContent(transaction);
            
            sendHtmlEmail(user.getEmail(), subject, htmlContent);
            log.info("Overdue notification sent to user: {}", user.getEmail());
            
        } catch (Exception e) {
            log.error("Failed to send overdue notification to user: {}", 
                    transaction.getUser().getEmail(), e);
        }
    }
    
    /**
     * Send simple text email
     */
    private void sendSimpleEmail(String to, String subject, String text) {
        SimpleMailMessage message = new SimpleMailMessage();
        message.setFrom(fromEmail);
        message.setTo(to);
        message.setSubject(subject);
        message.setText(text);
        
        mailSender.send(message);
    }
    
    /**
     * Send HTML email
     */
    private void sendHtmlEmail(String to, String subject, String htmlContent) throws MessagingException {
        MimeMessage message = mailSender.createMimeMessage();
        MimeMessageHelper helper = new MimeMessageHelper(message, true, "UTF-8");
        
        try {
            helper.setFrom(fromEmail, libraryName);
        } catch (Exception e) {
            // Fallback to simple from address if encoding fails
            helper.setFrom(fromEmail);
            log.warn("Failed to set from name, using simple email address: {}", e.getMessage());
        }
        
        helper.setTo(to);
        helper.setSubject(subject);
        helper.setText(htmlContent, true);
        
        mailSender.send(message);
    }
    
    /**
     * Build HTML content for book borrowed notification
     */
    private String buildBookBorrowedEmailContent(Transaction transaction) {
        return """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                    .header { background-color: #4CAF50; color: white; padding: 20px; text-align: center; }
                    .content { padding: 20px; background-color: #f9f9f9; }
                    .book-info { background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; }
                    .important { color: #d32f2f; font-weight: bold; }
                    .footer { text-align: center; padding: 20px; color: #666; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📚 Book Borrowed Successfully!</h1>
                    </div>
                    <div class="content">
                        <p>Dear <strong>%s</strong>,</p>
                        
                        <p>Congratulations! You have successfully borrowed the following book:</p>
                        
                        <div class="book-info">
                            <h3>📖 %s</h3>
                            <p><strong>Author:</strong> %s</p>
                            <p><strong>Transaction ID:</strong> %s</p>
                            <p><strong>Borrowed Date:</strong> %s</p>
                            <p class="important"><strong>Due Date:</strong> %s</p>
                        </div>
                        
                        <p><strong>Important Reminders:</strong></p>
                        <ul>
                            <li>Please return the book by the due date to avoid overdue fees</li>
                            <li>Overdue fee: <strong>5,000 VND per day</strong></li>
                            <li>You will receive a reminder email 3 days before the due date</li>
                            <li>Take good care of the book and return it in good condition</li>
                        </ul>
                        
                        <p>Thank you for using our library services!</p>
                    </div>
                    <div class="footer">
                        <p>%s<br>
                        This is an automated message, please do not reply.</p>
                    </div>
                </div>
            </body>
            </html>
            """.formatted(
                transaction.getUser().getUsername(),
                transaction.getBook().getTitle(),
                transaction.getBook().getAuthor(),
                transaction.getTransactionId(),
                transaction.getBorrowDate().format(DATE_FORMATTER),
                transaction.getDueDate().format(DATE_FORMATTER),
                libraryName
            );
    }
    
    /**
     * Build HTML content for return reminder notification
     */
    private String buildReturnReminderEmailContent(Transaction transaction) {
        return """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                    .header { background-color: #FF9800; color: white; padding: 20px; text-align: center; }
                    .content { padding: 20px; background-color: #f9f9f9; }
                    .book-info { background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #FF9800; }
                    .warning { color: #d32f2f; font-weight: bold; background-color: #ffebee; padding: 10px; border-radius: 5px; }
                    .footer { text-align: center; padding: 20px; color: #666; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>⏰ Return Reminder</h1>
                    </div>
                    <div class="content">
                        <p>Dear <strong>%s</strong>,</p>
                        
                        <p>This is a friendly reminder that your borrowed book is due for return in <strong>3 days</strong>.</p>
                        
                        <div class="book-info">
                            <h3>📖 %s</h3>
                            <p><strong>Author:</strong> %s</p>
                            <p><strong>Transaction ID:</strong> %s</p>
                            <p><strong>Borrowed Date:</strong> %s</p>
                            <p><strong>Due Date:</strong> %s</p>
                        </div>
                        
                        <div class="warning">
                            ⚠️ <strong>Important:</strong> Please return the book by the due date to avoid overdue fees of 5,000 VND per day.
                        </div>
                        
                        <p><strong>How to return:</strong></p>
                        <ul>
                            <li>Visit the library during operating hours</li>
                            <li>Return the book at the circulation desk</li>
                            <li>Get your return receipt for confirmation</li>
                        </ul>
                        
                        <p>Thank you for your cooperation!</p>
                    </div>
                    <div class="footer">
                        <p>%s<br>
                        This is an automated message, please do not reply.</p>
                    </div>
                </div>
            </body>
            </html>
            """.formatted(
                transaction.getUser().getUsername(),
                transaction.getBook().getTitle(),
                transaction.getBook().getAuthor(),
                transaction.getTransactionId(),
                transaction.getBorrowDate().format(DATE_FORMATTER),
                transaction.getDueDate().format(DATE_FORMATTER),
                libraryName
            );
    }
    
    /**
     * Build HTML content for overdue notification
     */
    private String buildOverdueEmailContent(Transaction transaction) {
        return """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                    .header { background-color: #f44336; color: white; padding: 20px; text-align: center; }
                    .content { padding: 20px; background-color: #f9f9f9; }
                    .book-info { background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #f44336; }
                    .urgent { color: #d32f2f; font-weight: bold; background-color: #ffebee; padding: 15px; border-radius: 5px; }
                    .footer { text-align: center; padding: 20px; color: #666; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚨 OVERDUE NOTICE</h1>
                    </div>
                    <div class="content">
                        <p>Dear <strong>%s</strong>,</p>
                        
                        <div class="urgent">
                            ⚠️ <strong>URGENT:</strong> Your borrowed book is now OVERDUE. Please return it immediately to avoid additional fees.
                        </div>
                        
                        <div class="book-info">
                            <h3>📖 %s</h3>
                            <p><strong>Author:</strong> %s</p>
                            <p><strong>Transaction ID:</strong> %s</p>
                            <p><strong>Borrowed Date:</strong> %s</p>
                            <p><strong>Due Date:</strong> %s</p>
                            <p><strong>Current Overdue Fee:</strong> %s VND</p>
                        </div>
                        
                        <p><strong>Immediate Action Required:</strong></p>
                        <ul>
                            <li>Return the book as soon as possible</li>
                            <li>Pay the overdue fee at the library</li>
                            <li>Contact the library if you have any issues</li>
                        </ul>
                        
                        <p><strong>Note:</strong> Overdue fees continue to accumulate at 5,000 VND per day until the book is returned.</p>
                        
                        <p>Please contact us immediately if you need assistance.</p>
                    </div>
                    <div class="footer">
                        <p>%s<br>
                        This is an automated message, please do not reply.</p>
                    </div>
                </div>
            </body>
            </html>
            """.formatted(
                transaction.getUser().getUsername(),
                transaction.getBook().getTitle(),
                transaction.getBook().getAuthor(),
                transaction.getTransactionId(),
                transaction.getBorrowDate().format(DATE_FORMATTER),
                transaction.getDueDate().format(DATE_FORMATTER),
                transaction.getOverdueFee(),
                libraryName
            );
    }
} 